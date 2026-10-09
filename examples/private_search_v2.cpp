#include "../include/private_hnsw_v2.h"
#include "io.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <chrono>
#include <numeric>
#include <cstring>

using namespace hnsw;

template<typename ElemType>
bool saveMatrix(const std::string& path, const std::shared_ptr<simplepir::MatrixT<ElemType>>& mat) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    uint64_t rows = mat->rows, cols = mat->cols;
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char*>(mat->data.data()),
              (size_t)rows * cols * sizeof(ElemType));
    return ofs.good();
}

template<typename ElemType>
std::shared_ptr<simplepir::MatrixT<ElemType>> loadMatrix(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return nullptr;
    uint64_t rows, cols;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    auto mat = std::make_shared<simplepir::MatrixT<ElemType>>(rows, cols);
    ifs.read(reinterpret_cast<char*>(mat->data.data()),
             (size_t)rows * cols * sizeof(ElemType));
    if (!ifs.good()) return nullptr;
    return mat;
}

bool savePIRCache(const std::string& indexPath,
                  const std::shared_ptr<EmbMatrix>& embShared,
                  const std::shared_ptr<EmbMatrix>& embHint,
                  const std::shared_ptr<NbrMatrix>& nbrShared,
                  const std::shared_ptr<NbrMatrix>& nbrHint) {
    std::string base = indexPath + ".pir_cache";
    bool ok = true;
    ok &= saveMatrix<EmbElem>(base + ".emb_shared", embShared);
    ok &= saveMatrix<EmbElem>(base + ".emb_hint", embHint);
    ok &= saveMatrix<NbrElem>(base + ".nbr_shared", nbrShared);
    ok &= saveMatrix<NbrElem>(base + ".nbr_hint", nbrHint);
    return ok;
}

struct PIRCache {
    std::shared_ptr<EmbMatrix> embShared, embHint;
    std::shared_ptr<NbrMatrix> nbrShared, nbrHint;
};

bool loadPIRCache(const std::string& indexPath, PIRCache& cache) {
    std::string base = indexPath + ".pir_cache";
    cache.embShared = loadMatrix<EmbElem>(base + ".emb_shared");
    cache.embHint = loadMatrix<EmbElem>(base + ".emb_hint");
    cache.nbrShared = loadMatrix<NbrElem>(base + ".nbr_shared");
    cache.nbrHint = loadMatrix<NbrElem>(base + ".nbr_hint");
    return cache.embShared && cache.embHint && cache.nbrShared && cache.nbrHint;
}

struct QrelsData {
    std::map<int, std::set<int>> qrels;
    std::vector<int> indexToQid;
    bool isValid = false;
};

QrelsData loadQrels(const std::string& dataDir) {
    QrelsData data;

    std::string qrelsPath = dataDir + "/qrels.dev.small.tsv";
    std::ifstream qrelsFile(qrelsPath);
    if (!qrelsFile) return data;

    std::string line;
    while (std::getline(qrelsFile, line)) {
        std::istringstream iss(line);
        int qid, dummy, pid, rel;
        if (iss >> qid >> dummy >> pid >> rel) {
            if (rel > 0) data.qrels[qid].insert(pid);
        }
    }
    std::cout << "  Loaded " << data.qrels.size() << " qrels entries from " << qrelsPath << std::endl;

    std::string queriesPath = dataDir + "/queries.dev.small.tsv";
    std::ifstream queriesFile(queriesPath);
    if (!queriesFile) return data;

    std::vector<int> qids;
    while (std::getline(queriesFile, line)) {
        std::istringstream iss(line);
        int qid;
        if (iss >> qid) qids.push_back(qid);
    }
    std::sort(qids.begin(), qids.end());
    data.indexToQid = qids;

    data.isValid = true;
    std::cout << "  Query mapping: " << qids.size() << " queries" << std::endl;
    return data;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <index.hnsw_q8> <query.fvecs> <groundtruth.ivecs> <output_file> [nq] [k] [ef] [ablation]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  index.hnsw_q8      : Quantized HNSW index (must have clustering)" << std::endl;
    std::cout << "  query.fvecs        : Query vectors file" << std::endl;
    std::cout << "  groundtruth.ivecs  : Ground truth file" << std::endl;
    std::cout << "  output_file        : Output file for statistics" << std::endl;
    std::cout << "  nq                 : Number of queries (default: all)" << std::endl;
    std::cout << "  k                  : Number of neighbors (default: 10)" << std::endl;
    std::cout << "  ef                 : Result-heap size (default: 50)" << std::endl;
    std::cout << "  ablation           : Ablation flags (comma-separated, default: none):" << std::endl;
    std::cout << "                       no_prune          - disable centroid pruning" << std::endl;
    std::cout << "                       no_topcand        - disable topCand (set to 0)" << std::endl;
    std::cout << "                       no_batch          - disable batch PIR" << std::endl;
    std::cout << "                       rounds_N          - set fixed search iterations to N" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  ./private_search_v2 <index> <query> <gt> <stats> 200 10 50" << std::endl;
    std::cout << "  ./private_search_v2 <index> <query> <gt> <stats> 200 10 50 no_prune" << std::endl;
    std::cout << "  ./private_search_v2 <index> <query> <gt> <stats> 200 10 50 rounds_6,no_batch" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string indexPath = argv[1];
    std::string queryPath = argv[2];
    std::string gtPath = argv[3];
    std::string outputFile = argv[4];

    int numQueries = (argc > 5) ? std::stoi(argv[5]) : -1;
    int k = (argc > 6) ? std::stoi(argv[6]) : 10;
    int ef = (argc > 7) ? std::stoi(argv[7]) : 50;
    std::string ablation = (argc > 8) ? argv[8] : "";

    bool ablationDisablePrune = false;
    bool ablationDisableTopCand = false;
    bool ablationDisableBatch = false;
    int ablationCentroidEntryK = -1;
    int ablationMaxStaleRounds = -1;
    int ablationFixedIterations = -1;

    auto parseOneFlag = [&](const std::string& flag) -> bool {
        if (flag == "no_prune") {
            ablationDisablePrune = true;
        } else if (flag == "no_topcand") {
            ablationDisableTopCand = true;
        } else if (flag == "no_batch") {
            ablationDisableBatch = true;
        } else if (flag.substr(0, 15) == "centroid_entry_") {
            ablationCentroidEntryK = std::stoi(flag.substr(15));
        } else if (flag.substr(0, 6) == "stale_") {
            ablationMaxStaleRounds = std::stoi(flag.substr(6));
        } else if (flag.substr(0, 7) == "rounds_") {
            ablationFixedIterations = std::stoi(flag.substr(7));
        } else {
            return false;
        }
        return true;
    };

    if (!ablation.empty()) {
        std::istringstream ss(ablation);
        std::string flag;
        while (std::getline(ss, flag, ',')) {
            if (!parseOneFlag(flag)) {
                std::cerr << "Unknown ablation flag: " << flag << std::endl;
                std::cerr << "Valid options: no_prune, no_topcand, no_batch, centroid_entry_K, stale_N, rounds_N" << std::endl;
                return 1;
            }
        }
        std::cout << "Ablation: " << ablation << std::endl;
    }

    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << outputFile << std::endl;
        return 1;
    }
    std::cout << "Output will be written to: " << outputFile << std::endl;

    auto printBoth = [&](const std::string& msg) {
        std::cout << msg;
        out << msg;
    };

    printBoth("========================================\n");
    printBoth("  Private HNSW Search V2 (Neighbor PIR)\n");
    printBoth("========================================\n\n");

    printBoth("[Step 1] Loading HNSW index: " + indexPath + "\n");

    QuantizerConfig cfg;
    cfg.bits = 8;

    HNSWQuantizedIndex index(0, 16, 200, cfg, DistanceType::L2);
    index.load(indexPath);

    std::cout << "Quantization mode: " << (index.quantizer.config.isUnsigned ? "unsigned" : "signed") << std::endl;

    if (!index.hasClusteringData()) {
        std::cerr << "Error: Index must have clustering data. Run buildClustering first." << std::endl;
        return 1;
    }

    int numClusters = index.getNumClusters();

    out << "=== Index Info ===" << std::endl;
    out << "Nodes: " << index.ntotal.load() << std::endl;
    out << "Dimension: " << index.d << std::endl;
    out << "M=" << index.M << ", M0=" << index.M0 << ", efConstruction=" << index.efConstruction << std::endl;
    out << "Entry point: " << index.entryPoint.load() << std::endl;
    out << "Max level: " << index.currentMaxLevel.load() << std::endl;
    out << "Clusters: " << numClusters << std::endl;
    out << std::endl;

    std::cout << "  Nodes: " << index.ntotal.load() << std::endl;
    std::cout << "  Dimension: " << index.d << std::endl;
    std::cout << "  Clusters: " << numClusters << std::endl;

    std::cout << std::endl;

    printBoth("[Step 2] Building Private HNSW V2 server...\n");

    auto buildStart = std::chrono::high_resolution_clock::now();
    PrivateHNSWServerV2 server;
    server.build(index);
    auto buildEnd = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double>(buildEnd - buildStart).count();

    out << "Server build time: " << std::fixed << std::setprecision(2) << buildTime << " s" << std::endl;

    const auto& config = server.getConfig();
    out << "\n=== Private HNSW V2 Config ===" << std::endl;
    out << "Embedding dim: " << config.embeddingDim << std::endl;
    out << "Num nodes: " << config.numNodes << std::endl;
    out << "Num clusters: " << config.numClusters << std::endl;
    out << "Max cluster size: " << config.maxClusterSize << std::endl;
    out << "Max neighbors: " << config.maxNeighbors << std::endl;
    out << std::endl;

    printBoth("[Step 3] Setting up PIR...\n");

    auto pirSetupStart = std::chrono::high_resolution_clock::now();

    std::shared_ptr<EmbMatrix> embSharedMatrix, embHint;
    std::shared_ptr<NbrMatrix> nbrSharedMatrix, nbrHint;

    PIRCache cache;
    if (loadPIRCache(indexPath, cache)) {
        std::cout << "  PIR cache found! Loading from disk..." << std::endl;
        embSharedMatrix = cache.embShared;
        embHint = cache.embHint;
        nbrSharedMatrix = cache.nbrShared;
        nbrHint = cache.nbrHint;
        std::cout << "  Embedding shared: " << embSharedMatrix->rows << " x " << embSharedMatrix->cols << std::endl;
        std::cout << "  Embedding hint: " << embHint->rows << " x " << embHint->cols << std::endl;
        std::cout << "  Neighbor shared: " << nbrSharedMatrix->rows << " x " << nbrSharedMatrix->cols << std::endl;
        std::cout << "  Neighbor hint: " << nbrHint->rows << " x " << nbrHint->cols << std::endl;

        std::cout << "  Setting up Embedding PIR server (compressed DB)..." << std::endl;
        server.setupEmbeddingPIR(embSharedMatrix, embHint);
        std::cout << "  Setting up Neighbor PIR server (compressed DB)..." << std::endl;
        server.setupNeighborPIR(nbrSharedMatrix, nbrHint);
    } else {
        std::cout << "  No PIR cache found. Computing from scratch..." << std::endl;

        std::cout << "  Generating embedding shared matrix..." << std::endl;
        embSharedMatrix = PrivateHNSWUtilsV2::generateEmbeddingSharedMatrix(
            server.getEmbeddingPIRParams());
        std::cout << "  Embedding shared matrix: " << embSharedMatrix->rows << " x " << embSharedMatrix->cols << std::endl;

        std::cout << "  Generating neighbor shared matrix..." << std::endl;
        nbrSharedMatrix = PrivateHNSWUtilsV2::generateNeighborSharedMatrix(
            server.getNeighborPIRParams());
        std::cout << "  Neighbor shared matrix: " << nbrSharedMatrix->rows << " x " << nbrSharedMatrix->cols << std::endl;

        std::cout << "  Setting up Embedding PIR..." << std::endl;
        embHint = server.setupEmbeddingPIR(embSharedMatrix);
        std::cout << "  Setting up Neighbor PIR..." << std::endl;
        nbrHint = server.setupNeighborPIR(nbrSharedMatrix);

        std::cout << "  Saving PIR cache to disk..." << std::endl;
        if (savePIRCache(indexPath, embSharedMatrix, embHint, nbrSharedMatrix, nbrHint)) {
            std::cout << "  PIR cache saved successfully." << std::endl;
        } else {
            std::cerr << "  WARNING: Failed to save PIR cache." << std::endl;
        }
    }

    auto pirSetupEnd = std::chrono::high_resolution_clock::now();
    double pirSetupTime = std::chrono::duration<double>(pirSetupEnd - pirSetupStart).count();

    uint64_t embHintBytes = embHint->rows * embHint->cols * sizeof(EmbElem);
    uint64_t nbrHintBytes = nbrHint->rows * nbrHint->cols * sizeof(NbrElem);

    out << "=== PIR Setup ===" << std::endl;
    out << "Setup time: " << std::fixed << std::setprecision(2) << pirSetupTime << " s" << std::endl;
    out << "Embedding PIR hint: " << embHint->rows << " x " << embHint->cols << std::endl;
    out << "Offline download (embedding hint): " << std::fixed << std::setprecision(2) << embHintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    out << "Neighbor PIR hint: " << nbrHint->rows << " x " << nbrHint->cols << std::endl;
    out << "Offline download (neighbor hint): " << std::fixed << std::setprecision(2) << nbrHintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    out << "Total offline download: " << std::fixed << std::setprecision(2) << (embHintBytes + nbrHintBytes) / 1024.0 / 1024.0 << " MB" << std::endl;
    out << std::endl;

    std::cout << "  Embedding hint: " << embHint->rows << " x " << embHint->cols << " ("
              << std::fixed << std::setprecision(2) << embHintBytes / 1024.0 / 1024.0 << " MB)" << std::endl;
    std::cout << "  Neighbor hint: " << nbrHint->rows << " x " << nbrHint->cols << " ("
              << std::fixed << std::setprecision(2) << nbrHintBytes / 1024.0 / 1024.0 << " MB)" << std::endl;
    std::cout << "  PIR setup time: " << std::fixed << std::setprecision(2) << pirSetupTime << " s" << std::endl;
    std::cout << std::endl;

    printBoth("[Step 4] Initializing client...\n");

    PrivateHNSWClientV2 client;
    PrivateHNSWConfigV2 clientConfig = server.getConfig();
    if (ablationDisablePrune) clientConfig.disableCentroidPrune = true;
    if (ablationDisableTopCand) clientConfig.topCand = 0;
    if (ablationDisableBatch) clientConfig.disableBatchPIR = true;
    if (ablationCentroidEntryK >= 0) clientConfig.centroidEntryK = ablationCentroidEntryK;
    if (ablationMaxStaleRounds >= 0) clientConfig.maxStaleRounds = ablationMaxStaleRounds;
    if (ablationFixedIterations >= 0) clientConfig.fixedSearchIterations = ablationFixedIterations;
    client.init(server.getMetadata(), clientConfig);
    client.initEmbeddingPIR(
        server.getEmbeddingPIRParams(), embSharedMatrix, embHint);
    client.initNeighborPIR(
        server.getNeighborPIRParams(), nbrSharedMatrix, nbrHint);

    if (!ablation.empty()) {
        std::string ablMsg = "  Ablation: " + ablation + "\n";
        printBoth(ablMsg);
    }
    std::cout << std::endl;

    printBoth("[Step 5] Loading queries...\n");

    int queryDim, nq;
    std::vector<float> queryFloats;
    if (queryPath.find(".bvecs") != std::string::npos) {
        queryFloats = readBvecs(queryPath, queryDim, nq);
    } else {
        queryFloats = readFvecs(queryPath, queryDim, nq);
    }

    int gtDim, gtN;
    auto gt = readIvecs(gtPath, gtDim, gtN);

    if (numQueries <= 0 || numQueries > std::min(nq, gtN)) {
        numQueries = std::min(nq, gtN);
    }

    out << "=== Query Info ===" << std::endl;
    out << "Query file: " << queryPath << std::endl;
    out << "Total queries: " << nq << " (dim=" << queryDim << ")" << std::endl;
    out << "Ground truth: " << gtN << " (top-" << gtDim << ")" << std::endl;
    out << "Queries to run: " << numQueries << std::endl;
    out << "k=" << k << ", ef=" << ef << std::endl;
    out << std::endl;

    std::cout << "  Loaded " << nq << " queries, using " << numQueries << std::endl;

    std::string queryDir = queryPath.substr(0, queryPath.find_last_of("/\\"));
    QrelsData qrelsData = loadQrels(queryDir);
    if (qrelsData.isValid) {
        std::cout << "  Qrels loaded: will compute MRR using relevance judgments" << std::endl;
    } else {
        std::cout << "  No qrels found: MRR will use NN ground truth" << std::endl;
    }
    std::cout << std::endl;

    printBoth("[Step 6] Quantizing queries...\n");

    std::vector<std::vector<uint8_t>> quantizedQueries(numQueries);
    for (int i = 0; i < numQueries; ++i) {
        quantizedQueries[i].resize(queryDim);
        index.quantizer.quantize(queryFloats.data() + (size_t)i * queryDim,
                                  quantizedQueries[i].data(), 1);
    }
    std::cout << std::endl;

    printBoth("[Step 7] Performing private search on " + std::to_string(numQueries) + " queries...\n");

    out << "\n=== Per-Query Statistics ===" << std::endl;

    std::vector<int> allRecalls;
    std::vector<double> allRR;
    std::vector<int> allPirQueries;
    std::vector<int> allEmbPirQueries;
    std::vector<int> allNbrPirQueries;
    std::vector<int> allClustersAccessed;
    std::vector<int> allClustersSkipped;
    std::vector<int> allNodesVisited;
    std::vector<int> allResultsFromNeighbor;
    std::vector<int> allResultsFromCluster;
    std::vector<double> allPirTimes;
    std::vector<double> allSearchTimes;
    std::vector<double> allOnlineTimes;

    std::vector<uint64_t> allEmbQueryBytes;
    std::vector<uint64_t> allEmbAnswerBytes;
    std::vector<uint64_t> allNbrQueryBytes;
    std::vector<uint64_t> allNbrAnswerBytes;

    PrivateSearchStatsV2 accumulatedStats;

    auto searchStart = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < numQueries; ++q) {
        PrivateSearchStatsV2 stats;
        auto results = client.searchWithStats(
            quantizedQueries[q].data(), k, ef, server, stats
        );

        std::set<int> gtSet;
        int gtK = std::min(k, gtDim);
        for (int j = 0; j < gtK; ++j) {
            gtSet.insert(gt[q * gtDim + j]);
        }

        int recall = 0;
        double rr = 0.0;
        int rank = 0;

        for (const auto& [nodeId, dist] : results) {
            rank++;
            if (gtSet.count(nodeId)) recall++;
            if (rr == 0.0) {
                if (qrelsData.isValid && q < (int)qrelsData.indexToQid.size()) {
                    int qid = qrelsData.indexToQid[q];
                    auto it = qrelsData.qrels.find(qid);
                    if (it != qrelsData.qrels.end() && it->second.count(nodeId)) {
                        rr = 1.0 / rank;
                    }
                } else {
                    if (gtSet.count(nodeId)) rr = 1.0 / rank;
                }
            }
        }

        allRecalls.push_back(recall);
        allRR.push_back(rr);
        allPirQueries.push_back(stats.pirQueryCount);
        allEmbPirQueries.push_back(stats.embeddingPirCount);
        allNbrPirQueries.push_back(stats.neighborPirCount);
        allClustersAccessed.push_back(stats.clustersAccessed);
        allClustersSkipped.push_back(stats.clustersSkipped);
        allNodesVisited.push_back(stats.nodesVisited);
        allResultsFromNeighbor.push_back(stats.resultsFromNeighbor);
        allResultsFromCluster.push_back(stats.resultsFromCluster);
        allPirTimes.push_back(stats.totalPirTimeMs);
        allSearchTimes.push_back(stats.totalSearchTimeMs);
        allOnlineTimes.push_back(stats.onlineTimeMs());

        allEmbQueryBytes.push_back(stats.embQueryBytes);
        allEmbAnswerBytes.push_back(stats.embAnswerBytes);
        allNbrQueryBytes.push_back(stats.nbrQueryBytes);
        allNbrAnswerBytes.push_back(stats.nbrAnswerBytes);

        accumulatedStats.precomputeHsTimeMs += stats.precomputeHsTimeMs;
        accumulatedStats.upperLayerSearchTimeMs += stats.upperLayerSearchTimeMs;
        accumulatedStats.embQueryGenTimeMs += stats.embQueryGenTimeMs;
        accumulatedStats.embServerTimeMs += stats.embServerTimeMs;
        accumulatedStats.embRecoverTimeMs += stats.embRecoverTimeMs;
        accumulatedStats.nbrQueryGenTimeMs += stats.nbrQueryGenTimeMs;
        accumulatedStats.nbrServerTimeMs += stats.nbrServerTimeMs;
        accumulatedStats.nbrRecoverTimeMs += stats.nbrRecoverTimeMs;
        accumulatedStats.nbrDecodeTimeMs += stats.nbrDecodeTimeMs;
        accumulatedStats.cacheWriteTimeMs += stats.cacheWriteTimeMs;
        accumulatedStats.candidateManageTimeMs += stats.candidateManageTimeMs;
        accumulatedStats.totalSearchTimeMs += stats.totalSearchTimeMs;
        accumulatedStats.embeddingPirCount += stats.embeddingPirCount;
        accumulatedStats.neighborPirCount += stats.neighborPirCount;
        accumulatedStats.embCommRounds += stats.embCommRounds;
        accumulatedStats.nbrCommRounds += stats.nbrCommRounds;
        if (q < 10) {
            out << "\nQuery " << q << ":" << std::endl;
            out << "  Total PIR queries: " << stats.pirQueryCount << std::endl;
            out << "    - Embedding PIR: " << stats.embeddingPirCount << std::endl;
            out << "    - Neighbor PIR: " << stats.neighborPirCount << std::endl;
            out << "  Clusters accessed: " << stats.clustersAccessed << std::endl;
            out << "  Clusters skipped: " << stats.clustersSkipped << std::endl;
            out << "  Nodes visited: " << stats.nodesVisited << std::endl;
            out << "  Results from neighbor: " << stats.resultsFromNeighbor << std::endl;
            out << "  Results from cluster: " << stats.resultsFromCluster << std::endl;
            out << "  Cache hits: " << stats.cacheHits << std::endl;
            out << "  PIR time: " << std::fixed << std::setprecision(2) << stats.totalPirTimeMs << " ms" << std::endl;
            out << "  Total time: " << std::fixed << std::setprecision(2) << stats.totalSearchTimeMs << " ms" << std::endl;
            out << "  Comm rounds: " << stats.embCommRounds + stats.nbrCommRounds
                << " (Emb=" << stats.embCommRounds << ", Nbr=" << stats.nbrCommRounds << ")" << std::endl;
            out << "  Recall@" << k << ": " << recall << "/" << k << std::endl;
        }

        if ((q + 1) % 10 == 0 || q == numQueries - 1) {
            std::cout << "\r  Processed " << (q + 1) << "/" << numQueries << " queries" << std::flush;
        }
    }
    std::cout << std::endl;

    auto searchEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(searchEnd - searchStart).count();

    int totalRecall = std::accumulate(allRecalls.begin(), allRecalls.end(), 0);
    double avgRecall = 100.0 * totalRecall / (numQueries * k);
    double avgPirQueries = std::accumulate(allPirQueries.begin(), allPirQueries.end(), 0.0) / numQueries;
    double avgEmbPir = std::accumulate(allEmbPirQueries.begin(), allEmbPirQueries.end(), 0.0) / numQueries;
    double avgNbrPir = std::accumulate(allNbrPirQueries.begin(), allNbrPirQueries.end(), 0.0) / numQueries;
    double avgClusters = std::accumulate(allClustersAccessed.begin(), allClustersAccessed.end(), 0.0) / numQueries;
    double avgClustersSkipped = std::accumulate(allClustersSkipped.begin(), allClustersSkipped.end(), 0.0) / numQueries;
    double avgNodes = std::accumulate(allNodesVisited.begin(), allNodesVisited.end(), 0.0) / numQueries;
    double avgResultsFromNeighbor = std::accumulate(allResultsFromNeighbor.begin(), allResultsFromNeighbor.end(), 0.0) / numQueries;
    double avgResultsFromCluster = std::accumulate(allResultsFromCluster.begin(), allResultsFromCluster.end(), 0.0) / numQueries;
    double avgPirTime = std::accumulate(allPirTimes.begin(), allPirTimes.end(), 0.0) / numQueries;
    double avgSearchTime = std::accumulate(allSearchTimes.begin(), allSearchTimes.end(), 0.0) / numQueries;
    double avgOnlineTime = std::accumulate(allOnlineTimes.begin(), allOnlineTimes.end(), 0.0) / numQueries;
    double minSearchTime = *std::min_element(allSearchTimes.begin(), allSearchTimes.end());
    std::vector<double> sortedSearchTimes(allSearchTimes);
    std::sort(sortedSearchTimes.begin(), sortedSearchTimes.end());
    double medianSearchTime = (numQueries % 2 == 1)
        ? sortedSearchTimes[numQueries / 2]
        : (sortedSearchTimes[numQueries / 2 - 1] + sortedSearchTimes[numQueries / 2]) / 2.0;
    double avgMRR = std::accumulate(allRR.begin(), allRR.end(), 0.0) / numQueries;

    uint64_t totalEmbQuery = std::accumulate(allEmbQueryBytes.begin(), allEmbQueryBytes.end(), 0ULL);
    uint64_t totalEmbAnswer = std::accumulate(allEmbAnswerBytes.begin(), allEmbAnswerBytes.end(), 0ULL);
    uint64_t totalNbrQuery = std::accumulate(allNbrQueryBytes.begin(), allNbrQueryBytes.end(), 0ULL);
    uint64_t totalNbrAnswer = std::accumulate(allNbrAnswerBytes.begin(), allNbrAnswerBytes.end(), 0ULL);

    double avgEmbQuery = (double)totalEmbQuery / numQueries;
    double avgEmbAnswer = (double)totalEmbAnswer / numQueries;
    double avgNbrQuery = (double)totalNbrQuery / numQueries;
    double avgNbrAnswer = (double)totalNbrAnswer / numQueries;

    double avgTotalQuery = avgEmbQuery + avgNbrQuery;
    double avgTotalAnswer = avgEmbAnswer + avgNbrAnswer;
    double avgTotalComm = avgTotalQuery + avgTotalAnswer;

    out << "\n========================================" << std::endl;
    out << "Summary (" << numQueries << " queries, k=" << k << ", ef=" << ef << ")" << std::endl;
    out << "========================================" << std::endl;
    if (!ablation.empty()) {
        out << "Ablation: " << ablation << std::endl;
    }
    out << std::endl;
    out << "--- Private Search V2 ---" << std::endl;
    std::string mrrLabel = qrelsData.isValid ? "MRR@" + std::to_string(k) + " (qrels)" : "MRR@" + std::to_string(k) + " (NN)";
    out << "Recall@" << k << ": " << std::fixed << std::setprecision(2) << avgRecall << "%" << std::endl;
    out << mrrLabel << ": " << std::fixed << std::setprecision(4) << avgMRR << std::endl;
    out << "Avg PIR queries: " << std::fixed << std::setprecision(1) << avgPirQueries << std::endl;
    out << "  - Embedding PIR: " << std::fixed << std::setprecision(1) << avgEmbPir << std::endl;
    out << "  - Neighbor PIR: " << std::fixed << std::setprecision(1) << avgNbrPir << std::endl;
    out << "Avg clusters accessed: " << std::fixed << std::setprecision(1) << avgClusters
        << " (" << std::setprecision(1) << 100.0 * avgClusters / config.numClusters << "% of total)" << std::endl;
    out << "Avg clusters skipped: " << std::fixed << std::setprecision(1) << avgClustersSkipped
        << " (centroid-based pruning)" << std::endl;
    out << "Avg nodes visited: " << std::fixed << std::setprecision(1) << avgNodes << std::endl;
    out << "  - Results from neighbor: " << std::fixed << std::setprecision(1) << avgResultsFromNeighbor << std::endl;
    out << "  - Results from cluster: " << std::fixed << std::setprecision(1) << avgResultsFromCluster << std::endl;
    out << "Avg PIR time: " << std::fixed << std::setprecision(2) << avgPirTime << " ms" << std::endl;
    out << "Avg search time: " << std::fixed << std::setprecision(2) << avgSearchTime << " ms" << std::endl;
    out << "Min search time: " << std::fixed << std::setprecision(2) << minSearchTime << " ms" << std::endl;
    out << "Median search time: " << std::fixed << std::setprecision(2) << medianSearchTime << " ms" << std::endl;
    out << "Total search time: " << std::fixed << std::setprecision(2) << totalTime / 1000.0 << " s" << std::endl;
    out << std::endl;

    out << "========================================" << std::endl;
    out << "Communication Cost (per ANN query)" << std::endl;
    out << "========================================" << std::endl;
    out << std::endl;

    out << "--- Embedding PIR ---" << std::endl;
    out << "Query (upload):   " << std::fixed << std::setprecision(2) << avgEmbQuery / 1024.0 << " KB" << std::endl;
    out << "Answer (download): " << std::fixed << std::setprecision(2) << avgEmbAnswer / 1024.0 << " KB" << std::endl;
    out << "Subtotal:         " << std::fixed << std::setprecision(2) << (avgEmbQuery + avgEmbAnswer) / 1024.0 << " KB" << std::endl;
    out << std::endl;

    out << "--- Neighbor PIR ---" << std::endl;
    out << "Query (upload):   " << std::fixed << std::setprecision(2) << avgNbrQuery / 1024.0 << " KB" << std::endl;
    out << "Answer (download): " << std::fixed << std::setprecision(2) << avgNbrAnswer / 1024.0 << " KB" << std::endl;
    out << "Subtotal:         " << std::fixed << std::setprecision(2) << (avgNbrQuery + avgNbrAnswer) / 1024.0 << " KB" << std::endl;
    out << std::endl;

    out << "--- Total ---" << std::endl;
    out << "Total Query (upload):   " << std::fixed << std::setprecision(2) << avgTotalQuery / 1024.0 << " KB" << std::endl;
    out << "Total Answer (download): " << std::fixed << std::setprecision(2) << avgTotalAnswer / 1024.0 << " KB" << std::endl;
    out << "Total Communication:    " << std::fixed << std::setprecision(2) << avgTotalComm / 1024.0 << " KB" << std::endl;
    out << "                        " << std::fixed << std::setprecision(2) << avgTotalComm / 1024.0 / 1024.0 << " MB" << std::endl;
    out << std::endl;

    out << "--- Communication Rounds (per ANN query) ---" << std::endl;
    out << "Avg Emb rounds: " << std::fixed << std::setprecision(1) << (double)accumulatedStats.embCommRounds / numQueries << std::endl;
    out << "Avg Nbr rounds: " << std::fixed << std::setprecision(1) << (double)accumulatedStats.nbrCommRounds / numQueries << std::endl;
    out << "Avg total rounds: " << std::fixed << std::setprecision(1) << (double)(accumulatedStats.embCommRounds + accumulatedStats.nbrCommRounds) / numQueries << std::endl;
    out << std::endl;

    out << "--- Offline Cost ---" << std::endl;
    out << "Embedding Hint: " << std::fixed << std::setprecision(2) << embHintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    out << "Neighbor Hint:  " << std::fixed << std::setprecision(2) << nbrHintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    out << "Total Hint:     " << std::fixed << std::setprecision(2) << (embHintBytes + nbrHintBytes) / 1024.0 / 1024.0 << " MB" << std::endl;
    out << std::endl;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Summary (" << numQueries << " queries)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Recall@" << k << ":  " << std::fixed << std::setprecision(2) << avgRecall << "%" << std::endl;
    std::cout << "  " << mrrLabel << ":  " << std::fixed << std::setprecision(4) << avgMRR << std::endl;
    std::cout << std::endl;
    std::cout << "  Neighbor hint: " << std::fixed << std::setprecision(2) << nbrHintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Avg PIR queries: " << std::fixed << std::setprecision(1) << avgPirQueries << std::endl;
    std::cout << "  Avg online time: " << std::fixed << std::setprecision(2) << avgSearchTime << " ms" << std::endl;
    std::cout << "  Min online time: " << std::fixed << std::setprecision(2) << minSearchTime << " ms" << std::endl;
    std::cout << "  Median online time: " << std::fixed << std::setprecision(2) << medianSearchTime << " ms" << std::endl;
    std::cout << "  Avg precompute: " << std::fixed << std::setprecision(2) << accumulatedStats.precomputeHsTimeMs / numQueries << " ms (可离线)" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << totalTime / 1000.0 << " s" << std::endl;
    std::cout << std::endl;
    std::cout << "  --- Communication per Query ---" << std::endl;
    std::cout << "  EmbPIR: Query=" << std::fixed << std::setprecision(2) << avgEmbQuery / 1024.0 << " KB, Answer=" << avgEmbAnswer / 1024.0 << " KB" << std::endl;
    std::cout << "  NbrPIR: Query=" << std::fixed << std::setprecision(2) << avgNbrQuery / 1024.0 << " KB, Answer=" << avgNbrAnswer / 1024.0 << " KB" << std::endl;
    std::cout << "  Total:  " << std::fixed << std::setprecision(2) << avgTotalComm / 1024.0 << " KB (" << avgTotalComm / 1024.0 / 1024.0 << " MB)" << std::endl;
    std::cout << std::endl;
    std::cout << "  --- Communication Rounds ---" << std::endl;
    std::cout << "  Avg Emb rounds: " << std::fixed << std::setprecision(1) << (double)accumulatedStats.embCommRounds / numQueries << std::endl;
    std::cout << "  Avg Nbr rounds: " << std::fixed << std::setprecision(1) << (double)accumulatedStats.nbrCommRounds / numQueries << std::endl;
    std::cout << "  Avg total rounds: " << std::fixed << std::setprecision(1) << (double)(accumulatedStats.embCommRounds + accumulatedStats.nbrCommRounds) / numQueries << std::endl;

    accumulatedStats.precomputeHsTimeMs /= numQueries;
    accumulatedStats.upperLayerSearchTimeMs /= numQueries;
    accumulatedStats.embQueryGenTimeMs /= numQueries;
    accumulatedStats.embServerTimeMs /= numQueries;
    accumulatedStats.embRecoverTimeMs /= numQueries;
    accumulatedStats.nbrQueryGenTimeMs /= numQueries;
    accumulatedStats.nbrServerTimeMs /= numQueries;
    accumulatedStats.nbrRecoverTimeMs /= numQueries;
    accumulatedStats.nbrDecodeTimeMs /= numQueries;
    accumulatedStats.cacheWriteTimeMs /= numQueries;
    accumulatedStats.candidateManageTimeMs /= numQueries;
    accumulatedStats.totalSearchTimeMs /= numQueries;
    accumulatedStats.embeddingPirCount /= numQueries;
    accumulatedStats.neighborPirCount /= numQueries;
    accumulatedStats.printDetailedTiming();

    std::cout << "Results written to: " << outputFile << std::endl;
    std::cout << "========================================" << std::endl;

    out.close();
    return 0;
}

#include "hnsw_quantized.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace hnsw;

float analyzeDataRange(const float* data, int n, int dim, bool* hasNegative = nullptr) {
    float minVal = data[0], maxVal = data[0];
    
    int sampleSize = std::min(n, 10000);
    int step = std::max(1, n / sampleSize);
    
    for (int i = 0; i < n; i += step) {
        for (int j = 0; j < dim; j++) {
            float v = data[(size_t)i * dim + j];
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
    }
    
    std::cout << "Data range: [" << minVal << ", " << maxVal << "]" << std::endl;

    if (hasNegative) *hasNegative = (minVal < 0);

    float absMax = std::max(std::abs(minVal), std::abs(maxVal));
    float scale;
    
    if (minVal >= 0) {
        scale = 255.0f / maxVal;
        std::cout << "Data type: non-negative, using unsigned quantization" << std::endl;
    } else {
        scale = 127.0f / absMax;
        std::cout << "Data type: signed, consider using signed quantization" << std::endl;
    }
    
    std::cout << "Recommended scale: " << scale << std::endl;
    return scale;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Quantized HNSW Index Builder (Parallel)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Usage: " << argv[0] << " <base.fvecs> <output.hnsw_q8> [M] [efConstruction] [scale] [threads] [nClusters] [graphPartition]" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  base.fvecs      : Input vectors in fvecs/bvecs format" << std::endl;
        std::cout << "  output.hnsw_q8  : Output quantized index file" << std::endl;
        std::cout << "  M               : Number of neighbors per node (default: 16)" << std::endl;
        std::cout << "  efConstruction  : Construction beam width (default: 200)" << std::endl;
        std::cout << "  scale           : Quantization scale factor (default: auto, set 0 for auto)" << std::endl;
        std::cout << "  threads         : Number of threads (default: auto, set 0 for auto)" << std::endl;
        std::cout << "  nClusters       : Number of clusters (default: sqrt(n/d))" << std::endl;
        std::cout << "  graphPartition  : Use graph partitioning (1=yes, 0=K-means, default: 1)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  ./build_index_quantized ../data/SIFT10K/siftsmall_base.fvecs ../data/index/siftsmall.hnsw_q8 32 200 0 4 -1 1" << std::endl;
        std::cout << std::endl;
        std::cout << "Output:" << std::endl;
        std::cout << "  <output.hnsw_q8> : HNSW index file (includes clustering data)" << std::endl;
        return 1;
    }
    
    std::string baseFile = argv[1];
    std::string indexFile = argv[2];
    int M = (argc > 3) ? std::stoi(argv[3]) : 16;
    int efConstruction = (argc > 4) ? std::stoi(argv[4]) : 200;
    float userScale = (argc > 5) ? std::stof(argv[5]) : 0.0f;
    int numThreads = (argc > 6) ? std::stoi(argv[6]) : 0;
    int nClusters = (argc > 7) ? std::stoi(argv[7]) : -1;
    bool useGraphPartition = (argc > 8) ? (std::stoi(argv[8]) != 0) : true;
    bool normalize = true;
    
    std::cout << "\n[1] Loading data from: " << baseFile << std::endl;
    
    int dim, n;
    std::vector<float> base;
    
    if (baseFile.find(".bvecs") != std::string::npos) {
        base = readBvecs(baseFile, dim, n);
    } else {
        base = readFvecs(baseFile, dim, n);
    }
    
    std::cout << "Loaded " << n << " vectors, dim=" << dim << std::endl;
    std::cout << "Data size: " << (size_t)n * dim * sizeof(float) / 1024.0 / 1024.0 << " MB (float32)" << std::endl;
    
    std::cout << "\n[2] Analyzing data range..." << std::endl;
    
    bool hasNegative = false;
    float scale = userScale;
    if (scale <= 0) {
        scale = analyzeDataRange(base.data(), n, dim, &hasNegative);
    } else {
        hasNegative = *std::min_element(base.begin(), base.end()) < 0;
        std::cout << "Using user-specified scale: " << scale << std::endl;
    }

    QuantizerConfig cfg;
    cfg.bits = 8;
    cfg.scale = scale;
    cfg.isUnsigned = !hasNegative;
    cfg.normalize = normalize;
    
    std::cout << "\nQuantization config:" << std::endl;
    std::cout << "  Bits: " << cfg.bits << std::endl;
    std::cout << "  Scale: " << cfg.scale << std::endl;
    std::cout << "  Mode: " << (cfg.isUnsigned ? "unsigned [0,255]" : "signed [-128,127]") << std::endl;
    std::cout << "  Normalize: " << (cfg.normalize ? "yes" : "no") << std::endl;
    
    std::string ckptGraph = indexFile + ".ckpt_graph";
    std::string ckptCluster = indexFile + ".ckpt_cluster";

    HNSWQuantizedIndex index(dim, M, efConstruction, cfg, DistanceType::L2);

    bool resumedFromCluster = false;
    bool resumedFromGraph = false;

    if (std::ifstream(ckptCluster).good()) {
        std::cout << "\n[Checkpoint] Found cluster checkpoint: " << ckptCluster << std::endl;
        std::cout << "  Resuming from clustering stage..." << std::endl;
        index.load(ckptCluster);
        n = index.ntotal.load();
        dim = index.d;
        resumedFromCluster = true;
    } else if (std::ifstream(ckptGraph).good()) {
        std::cout << "\n[Checkpoint] Found graph checkpoint: " << ckptGraph << std::endl;
        std::cout << "  Resuming from graph build stage..." << std::endl;
        index.load(ckptGraph);
        n = index.ntotal.load();
        dim = index.d;
        resumedFromGraph = true;
    }

    double buildTime = 0;
    if (!resumedFromCluster && !resumedFromGraph) {
        std::cout << "\n[3] Building HNSW graph..." << std::endl;
        std::cout << "Parameters: M=" << M << ", efConstruction=" << efConstruction
                  << ", threads=" << (numThreads > 0 ? std::to_string(numThreads) : "auto") << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();

#ifdef USE_HNSWLIB
        std::cout << "Using hnswlib for fast graph building..." << std::endl;
        index.buildWithHnswlibMove(std::move(base), n, numThreads);
#else
        std::cout << "Using built-in graph builder (float mode)..." << std::endl;
        index.addFloatParallelMove(std::move(base), n, numThreads);
        std::cout << "\n[3.5] Early quantization (free float vectors before clustering)..." << std::endl;
        index.finalizeQuantization();
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        buildTime = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\nGraph build completed!" << std::endl;
        std::cout << "  Build time: " << buildTime << "s" << std::endl;
        std::cout << "  Throughput: " << n / buildTime << " vectors/s" << std::endl;

        if (!std::getenv("SKIP_CHECKPOINT")) {
            std::cout << "  Saving graph checkpoint: " << ckptGraph << std::endl;
            index.save(ckptGraph);
            std::cout << "  Graph checkpoint saved." << std::endl;
        } else {
            std::cout << "  Skipping graph checkpoint (SKIP_CHECKPOINT=1)." << std::endl;
        }
    }

    if (!resumedFromCluster) {
        auto tPost0 = std::chrono::high_resolution_clock::now();
        std::cout << "\n[4] Building clustering..." << std::endl;

        if (useGraphPartition) {
            std::cout << "Using graph partitioning (better balance)..." << std::endl;
            index.buildGraphPartitionClustering(nClusters, 0.05);
        } else {
            std::cout << "Using K-means clustering..." << std::endl;
            index.buildClustering(nClusters);
        }

        std::cout << "\n[4.5] Building subgroup partitioning..." << std::endl;

        index.renumberNodesByCluster();

        int M0_val = 2 * M;
        int numClustersForCalc = index.getNumClusters();
        int maxClusterSizeForCalc = 0;
        for (int c = 0; c < numClustersForCalc; ++c) {
            int cSize = 0;
            for (int i = 0; i < n; ++i) {
                if (index.getClusterAssignment(i) == c) cSize++;
            }
            maxClusterSizeForCalc = std::max(maxClusterSizeForCalc, cSize);
        }
        int totalBitsCalc = (int)std::ceil(std::log2(numClustersForCalc + 1))
                          + (int)std::ceil(std::log2(maxClusterSizeForCalc + 1));
        int numPartsCalc = (totalBitsCalc + 6) / 7;
        int targetSubgroupSize = std::max(120, (int)std::sqrt((double)n / (M0_val * numPartsCalc)));
        std::cout << "  Dynamic targetSubgroupSize: " << targetSubgroupSize
                  << " (totalBits=" << totalBitsCalc << ", numParts=" << numPartsCalc << ")" << std::endl;
        index.buildSubgroups(targetSubgroupSize);

        index.buildNeighborInfo();

        auto tPost1 = std::chrono::high_resolution_clock::now();
        double postTime = std::chrono::duration<double>(tPost1 - tPost0).count();
        std::cout << "\n  Clustering + subgroup time: " << postTime << "s" << std::endl;
        if (!resumedFromGraph) {
            std::cout << "  Graph build total: " << (buildTime + postTime) << "s"
                      << " (HNSW=" << buildTime << "s, post=" << postTime << "s)" << std::endl;
        }

        if (!std::getenv("SKIP_CHECKPOINT")) {
            std::cout << "  Saving cluster checkpoint: " << ckptCluster << std::endl;
            index.save(ckptCluster);
            std::cout << "  Cluster checkpoint saved." << std::endl;
        } else {
            std::cout << "  Skipping cluster checkpoint (SKIP_CHECKPOINT=1)." << std::endl;
        }
    }

    std::cout << "\n[6] Saving final index to: " << indexFile << std::endl;

    index.save(indexFile);

    std::ifstream checkFile(indexFile, std::ios::binary | std::ios::ate);
    size_t savedSize = checkFile.tellg();
    size_t expectedMinSize = (size_t)n * dim + 1024;
    checkFile.close();
    if (savedSize > expectedMinSize) {
        std::remove(ckptGraph.c_str());
        std::remove(ckptCluster.c_str());
        std::cout << "  Index saved (" << savedSize / 1024.0 / 1024.0 << " MB). Checkpoints cleaned up." << std::endl;
    } else {
        std::cerr << "  WARNING: Saved index too small (" << savedSize << " bytes), keeping checkpoints!" << std::endl;
    }
    
    size_t quantizedMem = (size_t)n * dim * sizeof(uint8_t);
    size_t floatMem = (size_t)n * dim * sizeof(float);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Index Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total vectors: " << n << std::endl;
    std::cout << "Dimensions: " << dim << std::endl;
    std::cout << "Max level: " << index.currentMaxLevel.load() << std::endl;
    std::cout << "\nMemory usage:" << std::endl;
    std::cout << "  Quantized vectors: " << quantizedMem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Float32 equivalent: " << floatMem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Compression ratio: " << (float)floatMem / quantizedMem << "x" << std::endl;
    std::cout << "  Memory saving: " << (1.0 - (double)quantizedMem / floatMem) * 100 << "%" << std::endl;
    
    std::cout << "\nClustering info:" << std::endl;
    std::cout << "  Number of clusters: " << index.getNumClusters() << std::endl;
    std::cout << "  Method: " << (useGraphPartition ? "Graph Partitioning" : "K-means") << std::endl;

    auto clusterStats = index.analyzeCurrentClustering();
    std::cout << "  Imbalance: " << std::fixed << std::setprecision(2)
              << (clusterStats.imbalance * 100) << "%" << std::endl;
    std::cout << "  Max cluster size: " << clusterStats.maxClusterSize << std::endl;
    std::cout << "  Avg neighbor clusters: " << std::fixed << std::setprecision(2)
              << clusterStats.avgNeighborClusters << std::endl;
    std::cout << "  Edge cut ratio: " << std::fixed << std::setprecision(2)
              << (clusterStats.edgeCutRatio * 100) << "%" << std::endl;
    
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  Index (with clustering): " << indexFile << std::endl;
    std::cout << "\nUse 'private_search' or 'search_stats_quantized' to search this index." << std::endl;
    
    return 0;
}

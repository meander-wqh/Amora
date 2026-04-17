/**
 * @file private_hnsw_v2.cpp
 * @brief Implementation of privacy-preserving HNSW search with neighbor PIR
*/

#include "../include/private_hnsw_v2.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <chrono>
#include <climits>
#include <cstring>
#include <cmath>
#include <malloc.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hnsw {

// ============================================================================
// 无分支结果恢复辅助函数 (用于 Embedding PIR 内联恢复)
// ============================================================================
namespace {

// 无分支 clamp: 将 val 限制在 [0, maxVal] 范围内
inline int64_t branchlessClamp(int64_t val, int64_t maxVal) {
    int64_t negMask = val >> 63;
    val = val & ~negMask;  // val < 0 ? 0 : val

    int64_t diff = maxVal - val;
    int64_t overMask = diff >> 63;
    val = (val & ~overMask) | (maxVal & overMask);  // val > maxVal ? maxVal : val

    return val;
}

// 利用 EmbElem 类型自然溢出恢复 Embedding PIR 值
// EmbElem = uint32_t 时自动 mod 2^32，= uint64_t 时自动 mod 2^64
// 修改 EmbElem 类型时无需同步修改此函数
inline int64_t recoverEmbeddingValue(EmbElem ansVal, EmbElem hsVal, uint64_t delta) {
    EmbElem diff = ansVal - hsVal;  // 类型自然溢出，自动 mod 2^(sizeof(EmbElem)*8)

    int64_t signedDiff;
    if constexpr (sizeof(EmbElem) == sizeof(uint64_t)) {
        // 64位: 直接 reinterpret
        signedDiff = static_cast<int64_t>(diff);
    } else {
        // 32位或更小: 显式符号扩展
        constexpr uint64_t typeQ = 1ULL << (sizeof(EmbElem) * 8);
        constexpr uint64_t halfQ = typeQ / 2;
        if (static_cast<uint64_t>(diff) >= halfQ) {
            signedDiff = static_cast<int64_t>(diff) - static_cast<int64_t>(typeQ);
        } else {
            signedDiff = static_cast<int64_t>(diff);
        }
    }

    return (signedDiff + static_cast<int64_t>(delta / 2)) / static_cast<int64_t>(delta);
}

} // anonymous namespace

// ============================================================================
// PrivateHNSWConfigV2
// ============================================================================

void PrivateHNSWConfigV2::print() const {
    std::cout << "=== Private HNSW V2 Config ===" << std::endl;
    std::cout << "Embedding dim:     " << embeddingDim << std::endl;
    std::cout << "Num nodes:         " << numNodes << std::endl;
    std::cout << "Num clusters:      " << numClusters << std::endl;
    std::cout << "Max cluster size:  " << maxClusterSize << std::endl;
    std::cout << "Max neighbors:     " << maxNeighbors << std::endl;
    std::cout << "NeighborPIR logQ:  " << logQ << std::endl;
    std::cout << "Split encoding:    partBits=" << partBits << ", numParts=" << numParts << std::endl;
    std::cout << "Bit widths:        localBits=" << localBits << ", clusterBits=" << clusterBits << std::endl;
    std::cout << "Target subgroup:   " << targetSubgroupSize << std::endl;
    std::cout << "==============================" << std::endl;
}

// ============================================================================
// PrivateHNSWMetadataV2
// ============================================================================

void PrivateHNSWMetadataV2::print() const {
    std::cout << "=== Private HNSW V2 Metadata ===" << std::endl;
    std::cout << "Entry point:       " << entryPoint << std::endl;
    std::cout << "Entry cluster:     " << entryCluster << std::endl;
    std::cout << "Num clusters:      " << numClusters << std::endl;
    std::cout << "Max cluster size:  " << maxClusterSize << std::endl;
    std::cout << "Embedding dim:     " << embeddingDim << std::endl;
    std::cout << "Num nodes:         " << numNodes << std::endl;
    std::cout << "Max level:         " << maxLevel << std::endl;
    std::cout << "Upper layer nodes: " << upperLayerNodes.size() << std::endl;
    std::cout << "================================" << std::endl;
}

// ============================================================================
// PrivateSearchStatsV2
// ============================================================================

void PrivateSearchStatsV2::printDetailedTiming() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Detailed Timing Breakdown (per query)" << std::endl;
    std::cout << "========================================" << std::endl;

    double total = totalSearchTimeMs;
    auto pct = [total](double t) { return (total > 0) ? (t / total * 100.0) : 0.0; };

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\n[0] Precompute Hs (可异步/离线)" << std::endl;
    std::cout << "    Hs = Hint × secret:    " << std::setw(8) << precomputeHsTimeMs
              << " ms (可后台预计算，不计入在线时间)" << std::endl;

    std::cout << "\n[2] Upper Layer Search (no PIR)" << std::endl;
    std::cout << "    Upper search:         " << std::setw(8) << upperLayerSearchTimeMs
              << " ms (" << std::setw(5) << pct(upperLayerSearchTimeMs) << "%)" << std::endl;

    std::cout << "\n[3] Embedding PIR (per cluster, total " << embeddingPirCount << " queries)" << std::endl;
    std::cout << "    Query generation:      " << std::setw(8) << embQueryGenTimeMs
              << " ms (" << std::setw(5) << pct(embQueryGenTimeMs) << "%)" << std::endl;
    std::cout << "    Server compute:        " << std::setw(8) << embServerTimeMs
              << " ms (" << std::setw(5) << pct(embServerTimeMs) << "%)" << std::endl;
    std::cout << "    Result recovery:       " << std::setw(8) << embRecoverTimeMs
              << " ms (" << std::setw(5) << pct(embRecoverTimeMs) << "%)" << std::endl;
    std::cout << "    Subtotal:              " << std::setw(8) << totalEmbPirTimeMs()
              << " ms (" << std::setw(5) << pct(totalEmbPirTimeMs()) << "%)" << std::endl;

    std::cout << "\n[4] Neighbor PIR (per cluster, total " << neighborPirCount << " queries)" << std::endl;
    std::cout << "    Query generation:      " << std::setw(8) << nbrQueryGenTimeMs
              << " ms (" << std::setw(5) << pct(nbrQueryGenTimeMs) << "%)" << std::endl;
    std::cout << "    Server compute:        " << std::setw(8) << nbrServerTimeMs
              << " ms (" << std::setw(5) << pct(nbrServerTimeMs) << "%)" << std::endl;
    std::cout << "    Result recovery:       " << std::setw(8) << nbrRecoverTimeMs
              << " ms (" << std::setw(5) << pct(nbrRecoverTimeMs) << "%)" << std::endl;
    std::cout << "    Bitmap/ID decode:      " << std::setw(8) << nbrDecodeTimeMs
              << " ms (" << std::setw(5) << pct(nbrDecodeTimeMs) << "%)" << std::endl;
    std::cout << "    Subtotal:              " << std::setw(8) << totalNbrPirTimeMs()
              << " ms (" << std::setw(5) << pct(totalNbrPirTimeMs()) << "%)" << std::endl;

    std::cout << "\n[5] Other" << std::endl;
    std::cout << "    Candidate management:  " << std::setw(8) << candidateManageTimeMs
              << " ms (" << std::setw(5) << pct(candidateManageTimeMs) << "%)" << std::endl;
    std::cout << "    Cache write:           " << std::setw(8) << cacheWriteTimeMs
              << " ms (" << std::setw(5) << pct(cacheWriteTimeMs) << "%)" << std::endl;

    double unaccounted = total - onlineTimeMs();

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "  Online search time:      " << std::setw(8) << total << " ms" << std::endl;
    std::cout << "  Precompute (async):      " << std::setw(8) << precomputeHsTimeMs << " ms (不含在上述时间中)" << std::endl;
    std::cout << "  Unaccounted:             " << std::setw(8) << unaccounted << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// PrivateHNSWMetadataV2 辅助函数
// ============================================================================

int PrivateHNSWMetadataV2::getLocalIndexInSubgroup(int nodeId) const {
    if (!hasSubgrouping || nodeId < 0 || nodeId >= numNodes) return -1;

    if (nodeId >= (int)nodeToNewId.size()) return -1;
    int newId = nodeToNewId[nodeId];
    if (newId < 0) return -1;

    if (newId >= (int)nodeLocalIdxInSubgroup.size()) return -1;
    return nodeLocalIdxInSubgroup[newId];
}

std::pair<int, int> PrivateHNSWMetadataV2::getNodeSubgroup(int nodeId) const {
    if (!hasSubgrouping || nodeId < 0 || nodeId >= numNodes) {
        return {-1, -1};
    }

    int cluster = nodeToCluster[nodeId];

    if (nodeId >= (int)nodeToNewId.size()) {
        return {-1, -1};
    }
    int newId = nodeToNewId[nodeId];
    if (newId < 0 || newId >= (int)nodeSubgroup.size()) {
        return {-1, -1};
    }

    int subgroup = nodeSubgroup[newId];
    if (subgroup < 0) return {-1, -1};

    return {cluster, subgroup};
}

// ============================================================================
// PrivateHNSWServerV2
// ============================================================================

void PrivateHNSWServerV2::build(HNSWQuantizedIndex& index) {
    if (!index.hasClusteringData()) {
        throw std::runtime_error("Index must have clustering data");
    }

    int n = index.ntotal.load();
    int d = index.d;
    int numClusters = index.getNumClusters();

    std::cout << "[PrivateHNSWServerV2] Building from index..." << std::endl;
    std::cout << "  Nodes: " << n << ", Dim: " << d << ", Clusters: " << numClusters << std::endl;

    // Build metadata
    buildMetadata(index);
    std::cout << "  Metadata built." << std::endl;

    // Setup configuration
    config_.embeddingDim = d;
    config_.numNodes = n;
    config_.numClusters = numClusters;
    config_.maxClusterSize = metadata_.maxClusterSize;
    config_.maxNeighbors = index.M0;
    config_.targetSubgroupSize = metadata_.maxSubgroupSize;  // = targetSubgroupSize from build
    config_.isUnsigned = index.quantizer.config.isUnsigned;

    // Build Embedding PIR database
    simplepir::EmbeddingPIRConfig embConfig(d, numClusters, metadata_.maxClusterSize);
    embPirParams_.init(embConfig, config_.embeddingLogQ, config_.lweN);

    embDatabase_ = EmbDatabase(embConfig);

    bool signedQuant = !config_.isUnsigned;
    std::cout << "  Quantization mode: " << (signedQuant ? "signed (int8)" : "unsigned (uint8)") << std::endl;
    for (int c = 0; c < numClusters; ++c) {
        int csStart = metadata_.clusterOffset[c];
        int csEnd = metadata_.clusterOffset[c + 1];
        for (int newId = csStart; newId < csEnd; ++newId) {
            int oldId = metadata_.newIdToNode[newId];
            const uint8_t* vec = index.quantizedVectors.data() + (size_t)oldId * d;
            // 直接传入 uint8 原始 bit pattern，符号扩展由 matMulCompressed8_32 的 signedData 标志处理
            embDatabase_.addEmbeddingRaw(c, vec, d);
        }
    }
    embDatabase_.finalize();
    std::cout << "  Embedding PIR database built." << std::endl;

    // Build Neighbor PIR database
    if (metadata_.hasSubgrouping) {
        // 计算 maxSGPerCluster（统一列号: c * maxSGPerCluster + sg）
        config_.maxSGPerCluster = (metadata_.maxClusterSize + config_.targetSubgroupSize - 1) / config_.targetSubgroupSize;
        if (config_.maxSGPerCluster < 1) config_.maxSGPerCluster = 1;

        buildSubgroupNeighborPIRDatabase(index);
    } else {
        std::cerr << "[Warning] Index does not have subgroup info, neighbor PIR disabled." << std::endl;
        std::cerr << "         Please rebuild index with subgroup support." << std::endl;
    }

    isReady_ = true;
    std::cout << "[PrivateHNSWServerV2] Build complete." << std::endl;
}

void PrivateHNSWServerV2::buildMetadata(HNSWQuantizedIndex& index) {
    int n = index.ntotal.load();
    int numClusters = index.getNumClusters();
    int d = index.d;

    metadata_.numNodes = n;
    metadata_.numClusters = numClusters;
    metadata_.embeddingDim = d;
    metadata_.maxNeighbors = index.M0;
    metadata_.entryPoint = index.entryPoint.load();

    // Build cluster -> nodes mapping
    metadata_.clusterToNodes.resize(numClusters);
    metadata_.nodeToCluster.resize(n);
    metadata_.nodeIndexInCluster.resize(n);

    for (int i = 0; i < n; ++i) {
        int cluster = index.getClusterAssignment(i);
        metadata_.nodeToCluster[i] = cluster;
        metadata_.nodeIndexInCluster[i] = (int)metadata_.clusterToNodes[cluster].size();
        metadata_.clusterToNodes[cluster].push_back(i);
    }

    // Find max cluster size
    metadata_.maxClusterSize = 0;
    for (const auto& nodes : metadata_.clusterToNodes) {
        metadata_.maxClusterSize = std::max(metadata_.maxClusterSize, (int)nodes.size());
    }

    metadata_.entryCluster = metadata_.nodeToCluster[metadata_.entryPoint];

    // Extract all-level neighbors and node levels
    metadata_.allLevelNeighbors.resize(n);
    metadata_.nodeLevels.resize(n, 0);
    metadata_.maxLevel = index.currentMaxLevel.load();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int nodeLevel = (int)index.neighbors[i].size() - 1;
        if (nodeLevel < 0) nodeLevel = 0;
        metadata_.nodeLevels[i] = nodeLevel;

        metadata_.allLevelNeighbors[i].resize(nodeLevel + 1);

        for (int lev = 0; lev <= nodeLevel; ++lev) {
            if (lev < (int)index.neighbors[i].size()) {
                for (int32_t nbr : index.neighbors[i][lev]) {
                    if (nbr >= 0 && nbr < n) {
                        metadata_.allLevelNeighbors[i][lev].push_back(nbr);
                    }
                }
            }
        }
    }

    // Extract upper layer nodes and their embeddings
    metadata_.nodeToUpperIdx.resize(n, -1);
    for (int i = 0; i < n; ++i) {
        if (metadata_.nodeLevels[i] > 0) {
            int upperIdx = (int)metadata_.upperLayerNodes.size();
            metadata_.nodeToUpperIdx[i] = upperIdx;
            metadata_.upperLayerNodes.push_back(i);

            std::vector<uint8_t> emb(d);
            const uint8_t* src = index.quantizedVectors.data() + (size_t)i * d;
            std::copy(src, src + d, emb.begin());
            metadata_.upperLayerEmbeddings.push_back(std::move(emb));
        }
    }

    std::cout << "  Upper layer nodes: " << metadata_.upperLayerNodes.size() << std::endl;

    // ============================================
    // 计算聚类质心 (用于聚类级别剪枝)
    // ============================================
    metadata_.clusterCentroids.resize(numClusters);
    bool signedCentroid = !config_.isUnsigned;
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < numClusters; ++c) {
        std::vector<double> centroid(d, 0.0);
        const auto& nodes = metadata_.clusterToNodes[c];

        for (int nodeId : nodes) {
            const uint8_t* vec = index.quantizedVectors.data() + (size_t)nodeId * d;
            for (int j = 0; j < d; ++j) {
                if (signedCentroid) {
                    // uint8 位模式还原为 int8
                    centroid[j] += static_cast<int8_t>(vec[j]);
                } else {
                    centroid[j] += vec[j];
                }
            }
        }

        metadata_.clusterCentroids[c].resize(d);
        for (int j = 0; j < d; ++j) {
            double avg = centroid[j] / nodes.size();
            if (signedCentroid) {
                // int8 范围 [-128, 127]，存回 uint8 位模式
                int val = static_cast<int>(std::round(avg));
                val = std::max(-128, std::min(127, val));
                metadata_.clusterCentroids[c][j] = static_cast<uint8_t>(static_cast<int8_t>(val));
            } else {
                int val = static_cast<int>(avg + 0.5);
                metadata_.clusterCentroids[c][j] = static_cast<uint8_t>(std::min(255, std::max(0, val)));
            }
        }
    }
    std::cout << "  Cluster centroids computed." << std::endl;

    // ============================================
    // 加载子组信息 (使用 SubgroupManager)
    // ============================================
    if (index.hasSubgrouping) {
        subgroupManager_ = index.moveToSubgroupManager();

        metadata_.hasSubgrouping = true;
        metadata_.maxSubgroupSize = subgroupManager_.getMaxSubgroupSize();
        metadata_.totalSubgroups = subgroupManager_.getTotalSubgroups();

        // Move 数据避免深拷贝（build 后 subgroupManager_ 不再使用）
        metadata_.nodeToNewId = subgroupManager_.moveNodeToNewId();
        metadata_.newIdToNode = subgroupManager_.moveNewIdToNode();
        metadata_.clusterOffset = subgroupManager_.moveClusterOffsets();
        metadata_.clusterSubgroupOffset = subgroupManager_.moveSubgroupOffsets();
        metadata_.nodeSubgroup = subgroupManager_.moveNodeSubgroup();
        metadata_.nodeLocalIdxInSubgroup = subgroupManager_.moveNodeLocalIdx();

        std::cout << "    nodeSubgroup size: " << metadata_.nodeSubgroup.size() << std::endl;
        std::cout << "    nodeLocalIdxInSubgroup size: " << metadata_.nodeLocalIdxInSubgroup.size() << std::endl;

        std::cout << "    Sample nodeSubgroup values: ";
        for (int i = 0; i < std::min(10, (int)metadata_.nodeSubgroup.size()); ++i) {
            std::cout << metadata_.nodeSubgroup[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "    Sample nodeToNewId: ";
        for (int i = 0; i < std::min(10, (int)metadata_.nodeToNewId.size()); ++i) {
            std::cout << metadata_.nodeToNewId[i] << " ";
        }
        std::cout << std::endl;

        // 类型统一后直接 move，零拷贝
        metadata_.nodeNeighborGroups = subgroupManager_.moveAllNeighborInfo();

        std::cout << "  Subgroup info loaded (via SubgroupManager):" << std::endl;
        std::cout << "    Total subgroups: " << metadata_.totalSubgroups << std::endl;
        std::cout << "    Max subgroup size: " << metadata_.maxSubgroupSize << std::endl;
    } else {
        metadata_.hasSubgrouping = false;
        std::cout << "  No subgroup info in index." << std::endl;
    }
}


// ============================================================================
// 子组级 NeighborPIR 数据库构建 (使用 NeighborDatabaseT)
// ============================================================================
void PrivateHNSWServerV2::buildSubgroupNeighborPIRDatabase(const HNSWQuantizedIndex& index) {
    int C = metadata_.numClusters;
    int M0 = metadata_.maxNeighbors;
    int N = metadata_.numNodes;

    std::cout << "  Building Subgroup Neighbor PIR database (拆分编码)..." << std::endl;

    // 1. 计算动态位宽和拆分参数
    int localBits = (int)std::ceil(std::log2(metadata_.maxClusterSize + 1));
    int clusterBits = (int)std::ceil(std::log2(C + 1));
    int totalBits = clusterBits + localBits;
    int partBits = 7;
    int numParts = (totalBits + partBits - 1) / partBits;

    // 保存到 config
    config_.localBits = localBits;
    config_.clusterBits = clusterBits;
    config_.partBits = partBits;
    config_.numParts = numParts;

    std::cout << "    Encoding: clusterBits=" << clusterBits << " + localBits=" << localBits
              << " = totalBits=" << totalBits << ", numParts=" << numParts << std::endl;

    // PIR 参数
    int logP = partBits;  // 7
    uint64_t P = 1ULL << logP;  // 128
    uint32_t INVALID_PART = static_cast<uint32_t>(P - 1);  // 127

    // 2. 计算总列数（统一列号: c * maxSGPerCluster + sg）
    int maxSGPerCluster = config_.maxSGPerCluster;
    int maxSubgroupSize = metadata_.maxSubgroupSize;
    int totalCols_nb = C * maxSGPerCluster;

    // 也更新 metadata 中的旧映射（兼容性）
    metadata_.subgroupToColumn.resize(C);
    metadata_.numSubgroupsPerCluster.resize(C);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        int clusterSize = metadata_.clusterOffset[c + 1] - metadata_.clusterOffset[c];
        int numSG = (clusterSize + maxSubgroupSize - 1) / maxSubgroupSize;
        if (numSG < 1) numSG = 1;
        metadata_.numSubgroupsPerCluster[c] = numSG;
        metadata_.subgroupToColumn[c].resize(numSG);
        for (int g = 0; g < numSG; ++g) {
            metadata_.subgroupToColumn[c][g] = c * maxSGPerCluster + g;
        }
    }

    std::cout << "    Total columns (with padding): " << totalCols_nb << std::endl;

    // 3. 确定矩阵维度（行数包含 numParts 因子）
    int totalRows = maxSubgroupSize * M0 * numParts;

    std::cout << "    Matrix dimensions: " << totalRows << " × " << totalCols_nb << std::endl;
    std::cout << "    = (maxSubgroupSize=" << maxSubgroupSize << " × M0=" << M0
              << " × numParts=" << numParts << ") × totalCols" << std::endl;

    // 4. 创建 NeighborPIR 配置和数据库
    // Elem32 路径: 构造函数已用 INVALID_PART(127) 初始化，无需二次填充
    simplepir::NeighborPIRConfig nbrConfig(totalCols_nb, maxSubgroupSize, M0, numParts);
    nbrDatabase_ = NbrDatabase(nbrConfig);

    // 5. 填充邻居数据（拆分编码）— 按 newId 顺序迭代
    uint32_t partMask = (1u << partBits) - 1;  // 0x7F

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < C; ++c) {
        int csStart = metadata_.clusterOffset[c];
        int csEnd = metadata_.clusterOffset[c + 1];

        for (int localIdx = 0; localIdx < csEnd - csStart; ++localIdx) {
            int newId = csStart + localIdx;
            int nodeId = metadata_.newIdToNode[newId];
            if (nodeId < 0 || nodeId >= N) continue;

            // 计算节点所在子组和局部索引
            int subgroupInCluster = localIdx / maxSubgroupSize;
            int localIdxInSubgroup = localIdx % maxSubgroupSize;
            int colIdx = c * maxSGPerCluster + subgroupInCluster;

            if (nodeId >= (int)index.neighbors.size() || index.neighbors[nodeId].empty()) continue;

            const auto& neighbors = index.neighbors[nodeId][0];
            int numNeighbors = std::min((int)neighbors.size(), M0);

            for (int j = 0; j < numNeighbors; ++j) {
                int32_t nbrNodeId = neighbors[j];
                if (nbrNodeId < 0 || nbrNodeId >= N) continue;
                if (nbrNodeId >= (int)metadata_.nodeToNewId.size()) continue;
                int nbrNewId = metadata_.nodeToNewId[nbrNodeId];
                if (nbrNewId < 0) continue;

                int nbrCluster = metadata_.nodeToCluster[nbrNodeId];
                if (nbrCluster < 0 || nbrCluster >= C) continue;

                int nbrLocalInCluster = nbrNewId - metadata_.clusterOffset[nbrCluster];

                // 新编码: (clusterIdx << localBits) | localIdxInCluster
                uint32_t encoded = (static_cast<uint32_t>(nbrCluster) << localBits) |
                                   static_cast<uint32_t>(nbrLocalInCluster);

                // 拆分为 numParts 个部分，存储在连续行中
                for (int p = 0; p < numParts; ++p) {
                    uint32_t part = (encoded >> (p * partBits)) & partMask;
                    int row = localIdxInSubgroup * M0 * numParts + j * numParts + p;
                    nbrDatabase_.setElement(row, colIdx, static_cast<NbrElem>(part));
                }
            }
        }
    }

    // 6. 更新配置
    config_.useSubgroupNeighborPIR = true;
    config_.totalSubgroups = totalCols_nb;
    config_.maxSubgroupSize = maxSubgroupSize;

    // 7. 设置 PIR 参数（logQ=32, P=128）
    nbrPirParams_.init(nbrConfig, config_.logQ, config_.lweN, 6.4, P);

    // 8. 输出统计信息
    double matrixSizeMB = (double)totalRows * totalCols_nb * sizeof(NbrElem) / (1024 * 1024);
    double hintSizeMB = (double)totalRows * 1024 * sizeof(NbrElem) / (1024 * 1024);
    double queryPerPirKB = totalCols_nb * sizeof(NbrElem) / 1024.0;
    double answerPerPirKB = totalRows * sizeof(NbrElem) / 1024.0;

    std::cout << "    Subgroup Neighbor PIR database built (split encoding)." << std::endl;
    std::cout << "    logP=" << logP << ", P=" << P << ", numParts=" << numParts << std::endl;
    std::cout << "    Matrix size: " << std::fixed << std::setprecision(2) << matrixSizeMB << " MB" << std::endl;
    std::cout << "    Hint size: " << std::fixed << std::setprecision(2) << hintSizeMB << " MB" << std::endl;
    std::cout << "    Query/subgroup: " << std::fixed << std::setprecision(2) << queryPerPirKB << " KB" << std::endl;
    std::cout << "    Answer/subgroup: " << std::fixed << std::setprecision(2) << answerPerPirKB << " KB" << std::endl;
}

// ============================================================================
// 对称 PIR Setup
// ============================================================================

std::shared_ptr<EmbMatrix> PrivateHNSWServerV2::setupEmbeddingPIR(
    const std::shared_ptr<EmbMatrix>& sharedMatrix
) {
    // 设置有符号数据标志（影响压缩 matmul 的扩展方式）
    embPirServer_.setSignedData(!config_.isUnsigned);
    auto hint = embPirServer_.setup(embDatabase_, embPirParams_, sharedMatrix);
    // 释放原始 DB 矩阵（已压缩到 PIR server 内部的 compressedDB_）
    embDatabase_ = EmbDatabase();
    std::cout << "[PrivateHNSWServerV2] Embedding PIR setup complete. Hint: "
              << hint->rows << " x " << hint->cols << std::endl;
    return hint;
}

void PrivateHNSWServerV2::setupEmbeddingPIR(
    const std::shared_ptr<EmbMatrix>& sharedMatrix,
    const std::shared_ptr<EmbMatrix>& cachedHint
) {
    embPirServer_.setSignedData(!config_.isUnsigned);
    embPirServer_.setupWithCache(embDatabase_, embPirParams_, cachedHint);
    embDatabase_ = EmbDatabase();
    malloc_trim(0);
    std::cout << "[PrivateHNSWServerV2] Embedding PIR setup (cached). Hint: "
              << cachedHint->rows << " x " << cachedHint->cols << std::endl;
}

std::shared_ptr<NbrMatrix> PrivateHNSWServerV2::setupNeighborPIR(
    const std::shared_ptr<NbrMatrix>& sharedMatrix
) {
    auto hint = nbrPirServer_.setup(nbrDatabase_, nbrPirParams_, sharedMatrix);
    nbrDatabase_ = NbrDatabase();
    malloc_trim(0);
    std::cout << "[PrivateHNSWServerV2] Neighbor PIR setup complete. Hint: "
              << hint->rows << " x " << hint->cols << std::endl;
    return hint;
}

void PrivateHNSWServerV2::setupNeighborPIR(
    const std::shared_ptr<NbrMatrix>& sharedMatrix,
    const std::shared_ptr<NbrMatrix>& cachedHint
) {
    nbrPirServer_.setupWithCache(nbrDatabase_, nbrPirParams_, cachedHint);
    nbrDatabase_ = NbrDatabase();
    malloc_trim(0);
    std::cout << "[PrivateHNSWServerV2] Neighbor PIR setup (cached). Hint: "
              << cachedHint->rows << " x " << cachedHint->cols << std::endl;
}

// ============================================================================
// 对称 PIR Answer
// ============================================================================

EmbAnswerMsg PrivateHNSWServerV2::answerEmbeddingPIR(const EmbQueryMsg& query) const {
    return embPirServer_.answer(query);
}

std::shared_ptr<EmbMatrix> PrivateHNSWServerV2::batchAnswerEmbeddingPIR(
    const std::shared_ptr<EmbMatrix>& queryMatrix) const {
    return embPirServer_.batchAnswer(queryMatrix);
}

NbrAnswerMsg PrivateHNSWServerV2::answerNeighborPIR(const NbrQueryMsg& query) const {
    return nbrPirServer_.answer(query);
}

std::shared_ptr<NbrMatrix> PrivateHNSWServerV2::batchAnswerNeighborPIR(
    const std::shared_ptr<NbrMatrix>& queryMatrix) const {
    return nbrPirServer_.batchAnswer(queryMatrix);
}

// ============================================================================
// PrivateHNSWClientV2 — 对称初始化
// ============================================================================

void PrivateHNSWClientV2::init(
    const PrivateHNSWMetadataV2& metadata,
    const PrivateHNSWConfigV2& config
) {
    metadata_ = metadata;
    config_ = config;

    // 初始化优化后的缓存结构（按 newId 索引）
    distanceCache_.resize(metadata_.numNodes, INT64_MAX);
    neighborCache_.resize(metadata_.numNodes);
    cachedClusters_.resize(metadata_.numClusters, false);

    isReady_ = true;

    std::cout << "[PrivateHNSWClientV2] Initialized. "
              << metadata_.numClusters << " clusters, "
              << metadata_.numNodes << " nodes"
              << (config_.useSubgroupNeighborPIR ? ", subgroup mode" : "")
              << std::endl;
}

void PrivateHNSWClientV2::initEmbeddingPIR(
    const simplepir::EmbeddingPIRParams& params,
    const std::shared_ptr<EmbMatrix>& sharedMatrix,
    const std::shared_ptr<EmbMatrix>& hint
) {
    embPirParams_ = params;
    embPirClient_.init(params, sharedMatrix, hint);
}

void PrivateHNSWClientV2::initNeighborPIR(
    const simplepir::NeighborPIRParams& params,
    const std::shared_ptr<NbrMatrix>& sharedMatrix,
    const std::shared_ptr<NbrMatrix>& hint
) {
    nbrPirParams_ = params;
    nbrPirClient_.init(params, sharedMatrix, hint);
}

void PrivateHNSWClientV2::clearCache() {
    std::fill(distanceCache_.begin(), distanceCache_.end(), INT64_MAX);
    for (auto& nc : neighborCache_) {
        std::fill(nc.begin(), nc.end(), false);
    }
    std::fill(cachedClusters_.begin(), cachedClusters_.end(), false);

    neighborNodeIds_.clear();

    clusterCiphertextCache_.clear();
    totalRowsDecrypted_ = 0;
    totalRowsSkipped_ = 0;

    embPoolIdx_ = 0;
    nbrPoolIdx_ = 0;
}

// ============================================================================
// 预计算池优化
// ============================================================================

int PrivateHNSWClientV2::estimatePoolSize(int ef) const {
    int baseEstimate = std::min(ef + 5, metadata_.numClusters * 3 / 10);

    if (queryCount_ > 0 && avgEmbPoolUsage_ > 0) {
        int historyBased = static_cast<int>(avgEmbPoolUsage_ * 1.2);
        return std::max(baseEstimate, historyBased);
    }

    return baseEstimate;
}

void PrivateHNSWClientV2::precomputeSecretPool(int poolSize) {
    auto start = std::chrono::high_resolution_clock::now();

    int embPoolSize = poolSize;
    // 每次 NbrPIR 查询只消耗 1 个预计算对，不需要乘以 maxNeighbors
    int nbrPoolSize = poolSize;
    if (queryCount_ > 0 && avgNbrPoolUsage_ > 0) {
        nbrPoolSize = std::max(nbrPoolSize, static_cast<int>(avgNbrPoolUsage_ * 1.2));
    }

    // ============================================
    // 预分配查询向量内存池
    // ============================================
    uint64_t embQueryDim = embPirParams_.config.queryDim();
    uint64_t embLogQ = embPirParams_.pirParams.Logq;
    uint64_t nbrM = nbrPirParams_.pirParams.M;
    uint64_t nbrLogQ = nbrPirParams_.pirParams.Logq;

    embQueryPool_.resize(embPoolSize);
    nbrQueryPool_.resize(nbrPoolSize);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < embPoolSize; ++i) {
        embQueryPool_[i] = std::make_shared<EmbMatrix>(embQueryDim, 1, embLogQ);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < nbrPoolSize; ++i) {
        nbrQueryPool_[i] = std::make_shared<NbrMatrix>(nbrM, 1, nbrLogQ);
    }

    embQueryPoolIdx_ = 0;
    nbrQueryPoolIdx_ = 0;

    // ============================================
    // 预计算 Embedding PIR 池
    // ============================================
    embPool_.resize(embPoolSize);
    uint64_t embN = embPirParams_.pirParams.N;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < embPoolSize; ++i) {
        auto secret = EmbMatrix::random(embN, 1, embLogQ, 0);
        embPool_[i].Hs = simplepir::matrixMulVec(embPirClient_.getHint(), secret);
        embPool_[i].As = simplepir::matrixMulVec(embPirClient_.getSharedMatrix(), secret);
    }

    // ============================================
    // 预计算 Neighbor PIR 池
    // ============================================
    nbrPool_.resize(nbrPoolSize);
    uint64_t nbrN = nbrPirParams_.pirParams.N;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < nbrPoolSize; ++i) {
        auto secret = NbrMatrix::random(nbrN, 1, nbrLogQ, 0);
        nbrPool_[i].Hs = simplepir::matrixMulVec(nbrPirClient_.getHint(), secret);
        nbrPool_[i].As = simplepir::matrixMulVec(nbrPirClient_.getSharedMatrix(), secret);
    }

    embPoolIdx_ = 0;
    nbrPoolIdx_ = 0;
}

EmbPIRPrecompute PrivateHNSWClientV2::getEmbPrecompute() {
    if (embPoolIdx_ < embPool_.size()) {
        return embPool_[embPoolIdx_++];
    }

    EmbPIRPrecompute result;
    uint64_t N = embPirParams_.pirParams.N;
    uint64_t logQ = embPirParams_.pirParams.Logq;

    auto secret = EmbMatrix::random(N, 1, logQ, 0);
    result.Hs = simplepir::matrixMulVec(embPirClient_.getHint(), secret);
    result.As = simplepir::matrixMulVec(embPirClient_.getSharedMatrix(), secret);

    return result;
}

NbrPIRPrecompute PrivateHNSWClientV2::getNbrPrecompute() {
    if (nbrPoolIdx_ < nbrPool_.size()) {
        return nbrPool_[nbrPoolIdx_++];
    }

    NbrPIRPrecompute result;
    uint64_t N = nbrPirParams_.pirParams.N;
    uint64_t logQ = nbrPirParams_.pirParams.Logq;

    auto secret = NbrMatrix::random(N, 1, logQ, 0);
    result.Hs = simplepir::matrixMulVec(nbrPirClient_.getHint(), secret);
    result.As = simplepir::matrixMulVec(nbrPirClient_.getSharedMatrix(), secret);

    return result;
}

std::shared_ptr<EmbMatrix> PrivateHNSWClientV2::getEmbQueryBuffer() {
    if (embQueryPoolIdx_ < embQueryPool_.size()) {
        return embQueryPool_[embQueryPoolIdx_++];
    }
    uint64_t queryDim = embPirParams_.config.queryDim();
    return std::make_shared<EmbMatrix>(queryDim, 1, embPirParams_.pirParams.Logq);
}

std::shared_ptr<NbrMatrix> PrivateHNSWClientV2::getNbrQueryBuffer() {
    if (nbrQueryPoolIdx_ < nbrQueryPool_.size()) {
        return nbrQueryPool_[nbrQueryPoolIdx_++];
    }
    return std::make_shared<NbrMatrix>(nbrPirParams_.pirParams.M, 1, nbrPirParams_.pirParams.Logq);
}

bool PrivateHNSWClientV2::isClusterCached(int clusterId) const {
    return cachedClusters_[clusterId];
}

int64_t PrivateHNSWClientV2::getCachedDistance(int newId) const {
    return distanceCache_[newId];
}

bool PrivateHNSWClientV2::hasNeighborInCluster(int newId, int clusterId) const {
    if (neighborCache_[newId].empty()) return false;
    return neighborCache_[newId][clusterId];
}


void PrivateHNSWClientV2::batchQueryClusters(
    const std::vector<int>& clusterIds,
    const uint8_t* query,
    PrivateHNSWServerV2& server,
    PrivateSearchStatsV2& stats
) {
    int N = (int)clusterIds.size();
    if (N == 0) return;

    // 消融: 关闭批量 PIR 时逐个查询（通信字节不变，轮次从 1 变为 N）
    if (config_.disableBatchPIR || N == 1) {
        for (int i = 0; i < N; ++i) {
            queryCluster(clusterIds[i], query, server, stats);
        }
        return;
    }

    int d = metadata_.embeddingDim;
    int K = metadata_.maxClusterSize;
    int C = metadata_.numClusters;
    uint64_t queryDim = embPirParams_.config.queryDim();
    uint64_t delta = embPirParams_.delta();

    // 量化查询向量（只做一次）
    std::vector<uint64_t> queryEmb(d);
    bool signedQuant = !config_.isUnsigned;
    for (int i = 0; i < d; ++i) {
        if (signedQuant) {
            // 符号扩展: uint8 → int8 → int32 → uint32 → uint64
            queryEmb[i] = static_cast<uint64_t>(static_cast<uint32_t>(
                static_cast<int32_t>(static_cast<int8_t>(query[i]))));
        } else {
            queryEmb[i] = static_cast<uint64_t>(query[i]);
        }
    }

    // ============================================
    // 1. 获取 N 个预计算对，构建批量查询矩阵 Q (queryDim × N)
    // ============================================
    auto embQueryGenStart = std::chrono::high_resolution_clock::now();

    auto batchQuery = std::make_shared<EmbMatrix>(queryDim, N);
    std::vector<std::shared_ptr<EmbMatrix>> hsVec(N);

    // 一次生成所有噪声 (queryDim × N)
    auto noise = EmbMatrix::gaussian(queryDim, N);
    const EmbElem* noiseData = noise->data.data();
    EmbElem* bqData = batchQuery->data.data();
    uint64_t embDim = embPirParams_.d();

    for (int j = 0; j < N; ++j) {
        auto embPrecompute = getEmbPrecompute();
        hsVec[j] = embPrecompute.Hs;

        const EmbElem* asData = embPrecompute.As->data.data();

        // 合并 As 拷贝 + 噪声（直接操作 data 数组，跳过边界检查）
        for (uint64_t r = 0; r < queryDim; ++r) {
            bqData[r * N + j] = asData[r] + noiseData[r * N + j];
        }

        // 在目标聚类位置添加缩放的查询向量
        int clusterId = clusterIds[j];
        uint64_t colOffset = clusterId * embDim;
        for (uint64_t i = 0; i < embDim; ++i) {
            bqData[(colOffset + i) * N + j] += static_cast<EmbElem>(delta * queryEmb[i]);
        }
    }

    auto embQueryGenEnd = std::chrono::high_resolution_clock::now();
    stats.embQueryGenTimeMs += std::chrono::duration<double, std::milli>(embQueryGenEnd - embQueryGenStart).count();

    // ============================================
    // 2. 一次服务端计算: Ans = DB × Q → K × N 矩阵
    // ============================================
    auto embServerStart = std::chrono::high_resolution_clock::now();
    auto batchAns = server.batchAnswerEmbeddingPIR(batchQuery);
    auto embServerEnd = std::chrono::high_resolution_clock::now();
    stats.embServerTimeMs += std::chrono::duration<double, std::milli>(embServerEnd - embServerStart).count();

    // ============================================
    // 3. 逐列恢复 + 缓存写入
    // ============================================
    auto embRecoverStart = std::chrono::high_resolution_clock::now();

    // 预分配内积缓冲区，避免循环内重复分配
    if ((int)innerProductsBuf_.size() < K)
        innerProductsBuf_.resize(K);

    const EmbElem* ansData = batchAns->data.data();

    for (int j = 0; j < N; ++j) {
        int clusterId = clusterIds[j];
        auto embHs = hsVec[j];
        const EmbElem* hsData = embHs->data.data();

        // 恢复内积（直接操作 data 数组，跳过边界检查）
        for (int i = 0; i < K; ++i) {
            innerProductsBuf_[i] = recoverEmbeddingValue(
                ansData[i * N + j], hsData[i], delta);
        }

        // 密文缓存（与 queryCluster 一致）
        if (metadata_.hasSubgrouping) {
            auto& cache = clusterCiphertextCache_[clusterId];
            // 为密文缓存提取第 j 列为独立向量（直接操作 data 数组）
            auto colAns = std::make_shared<EmbMatrix>(K, 1);
            EmbElem* dst = colAns->data.data();
            for (int i = 0; i < K; ++i) {
                dst[i] = ansData[i * N + j];
            }
            cache.answer = colAns;
            cache.Hs = embHs;

            int numSubgroups = config_.maxSGPerCluster;
            cache.decryptedSubgroups.resize(numSubgroups, false);
            cache.decryptedDistances.resize(K, INT64_MAX);

            for (int i = 0; i < K; ++i) {
                cache.decryptedDistances[i] = innerProductsBuf_[i];
            }
            for (int g = 0; g < numSubgroups; ++g) {
                cache.decryptedSubgroups[g] = true;
            }
        }

        // 距离缓存写入（按 newId 索引）
        cachedClusters_[clusterId] = true;
        int csStart = metadata_.clusterOffset[clusterId];
        int csEnd = metadata_.clusterOffset[clusterId + 1];
        int clusterSize = csEnd - csStart;
        for (int k = 0; k < clusterSize; ++k) {
            int newId = csStart + k;
            distanceCache_[newId] = -innerProductsBuf_[k];
            if (neighborCache_[newId].empty()) {
                neighborCache_[newId].resize(C, false);
            }
        }
    }

    auto embRecoverEnd = std::chrono::high_resolution_clock::now();
    stats.embRecoverTimeMs += std::chrono::duration<double, std::milli>(embRecoverEnd - embRecoverStart).count();

    auto embRecoverEnd2 = std::chrono::high_resolution_clock::now();

    // ============================================
    // 4. 统计更新
    // ============================================
    stats.embeddingPirCount += N;
    stats.embCommRounds += 1;  // 批量 = 1 轮通信
    stats.pirQueryCount += N;
    stats.totalPirTimeMs += std::chrono::duration<double, std::milli>(embRecoverEnd2 - embQueryGenStart).count();

    // 通信量统计
    uint64_t embElemSize = sizeof(EmbElem);
    stats.embQueryBytes += queryDim * N * embElemSize;
    stats.embAnswerBytes += K * N * embElemSize;
}

ClusterPIRResult PrivateHNSWClientV2::queryCluster(
    int clusterId,
    const uint8_t* query,
    PrivateHNSWServerV2& server,
    PrivateSearchStatsV2& stats
) {
    ClusterPIRResult result;
    int d = metadata_.embeddingDim;
    int K = metadata_.maxClusterSize;
    int C = metadata_.numClusters;

    auto pirStart = std::chrono::high_resolution_clock::now();

    // ============================================
    // 1. Embedding PIR query
    // ============================================
    auto embQueryGenStart = std::chrono::high_resolution_clock::now();

    std::vector<uint64_t> queryEmb(d);
    bool signedQuant2 = !config_.isUnsigned;
    for (int i = 0; i < d; ++i) {
        if (signedQuant2) {
            queryEmb[i] = static_cast<uint64_t>(static_cast<uint32_t>(
                static_cast<int32_t>(static_cast<int8_t>(query[i]))));
        } else {
            queryEmb[i] = static_cast<uint64_t>(query[i]);
        }
    }

    uint64_t queryDim = embPirParams_.config.queryDim();

    auto embPrecompute = getEmbPrecompute();
    auto embHs = embPrecompute.Hs;

    auto embQueryVec = getEmbQueryBuffer();
    std::memcpy(embQueryVec->data.data(), embPrecompute.As->data.data(),
                queryDim * sizeof(EmbElem));

    auto noise = EmbMatrix::gaussian(queryDim, 1);
    embQueryVec->matrixAdd(*noise);

    uint64_t delta = embPirParams_.delta();
    uint64_t colOffset = clusterId * embPirParams_.d();
    for (uint64_t i = 0; i < embPirParams_.d(); ++i) {
        uint64_t currentVal = embQueryVec->get(colOffset + i, 0);
        uint64_t addition = delta * queryEmb[i];
        embQueryVec->set(colOffset + i, 0, currentVal + addition);
    }

    EmbQueryMsg embQueryMsg;
    embQueryMsg.queryVector = embQueryVec;
    embQueryMsg.batchSize = 1;

    auto embQueryGenEnd = std::chrono::high_resolution_clock::now();
    stats.embQueryGenTimeMs += std::chrono::duration<double, std::milli>(embQueryGenEnd - embQueryGenStart).count();

    auto embServerStart = std::chrono::high_resolution_clock::now();
    auto embAnswer = server.answerEmbeddingPIR(embQueryMsg);
    auto embServerEnd = std::chrono::high_resolution_clock::now();
    stats.embServerTimeMs += std::chrono::duration<double, std::milli>(embServerEnd - embServerStart).count();

    // 恢复内积（利用 EmbElem 类型自然溢出，切换类型时无需修改此处）
    auto embRecoverStart = std::chrono::high_resolution_clock::now();
    result.innerProducts.resize(K);

    if (metadata_.hasSubgrouping) {
        auto& cache = clusterCiphertextCache_[clusterId];
        cache.answer = embAnswer.answer;
        cache.Hs = embHs;

        int numSubgroups = config_.maxSGPerCluster;
        cache.decryptedSubgroups.resize(numSubgroups, false);
        cache.decryptedDistances.resize(K, INT64_MAX);

        totalRowsSkipped_ += K;

        for (int i = 0; i < K; ++i) {
            int64_t rounded = recoverEmbeddingValue(
                embAnswer.answer->get(i, 0), embHs->get(i, 0), delta);
            result.innerProducts[i] = rounded;
            cache.decryptedDistances[i] = rounded;
        }

        for (int g = 0; g < numSubgroups; ++g) {
            cache.decryptedSubgroups[g] = true;
        }
        totalRowsDecrypted_ += K;
        totalRowsSkipped_ -= K;
    } else {
        for (int i = 0; i < K; ++i) {
            result.innerProducts[i] = recoverEmbeddingValue(
                embAnswer.answer->get(i, 0), embHs->get(i, 0), delta);
        }
    }

    auto embRecoverEnd = std::chrono::high_resolution_clock::now();
    stats.embRecoverTimeMs += std::chrono::duration<double, std::milli>(embRecoverEnd - embRecoverStart).count();

    stats.embeddingPirCount++;
    stats.embCommRounds += 1;

    // ============================================
    // 2. Neighbor PIR query
    // ============================================
    if (config_.useSubgroupNeighborPIR) {
        result.neighborClusters.resize(K);
        for (int k = 0; k < K; ++k) {
            result.neighborClusters[k].resize(C, false);
        }

        auto pirEnd = std::chrono::high_resolution_clock::now();
        stats.totalPirTimeMs += std::chrono::duration<double, std::milli>(pirEnd - pirStart).count();
        stats.pirQueryCount++;

        uint64_t embElemSize = sizeof(EmbElem);
        uint64_t embQuerySize = embPirParams_.config.queryDim() * embElemSize;
        uint64_t embAnswerSize = K * embElemSize;
        stats.embQueryBytes += embQuerySize;
        stats.embAnswerBytes += embAnswerSize;

        auto cacheStart = std::chrono::high_resolution_clock::now();
        cachedClusters_[clusterId] = true;
        int csStart = metadata_.clusterOffset[clusterId];
        int csEnd = metadata_.clusterOffset[clusterId + 1];
        int clusterSize = csEnd - csStart;
        for (int k = 0; k < clusterSize; ++k) {
            int newId = csStart + k;
            distanceCache_[newId] = -result.innerProducts[k];
            if (neighborCache_[newId].empty()) {
                neighborCache_[newId].resize(C, false);
            }
        }
        auto cacheEnd = std::chrono::high_resolution_clock::now();
        stats.cacheWriteTimeMs += std::chrono::duration<double, std::milli>(cacheEnd - cacheStart).count();

        return result;
    }

    std::cerr << "[ERROR] queryCluster: useSubgroupNeighborPIR is false, unsupported." << std::endl;
    return result;
}

int PrivateHNSWClientV2::privateGreedySearch(
    const uint8_t* query,
    int entryPoint,
    int targetLevel
) {
    int d = metadata_.embeddingDim;
    int currNode = entryPoint;

    auto computeUpperLayerDist = [&](int nodeId) -> int64_t {
        int upperIdx = metadata_.nodeToUpperIdx[nodeId];
        if (upperIdx < 0) return INT64_MAX;

        const auto& emb = metadata_.upperLayerEmbeddings[upperIdx];
        int64_t ip = 0;
        for (int i = 0; i < d; ++i) {
            ip += (int64_t)query[i] * (int64_t)emb[i];
        }
        return -ip;
    };

    int64_t currDist = computeUpperLayerDist(currNode);

    for (int level = metadata_.maxLevel; level > targetLevel; --level) {
        bool changed = true;
        while (changed) {
            changed = false;

            if (currNode >= (int)metadata_.allLevelNeighbors.size()) break;
            if (level >= (int)metadata_.allLevelNeighbors[currNode].size()) break;

            for (int neighbor : metadata_.allLevelNeighbors[currNode][level]) {
                if (metadata_.nodeLevels[neighbor] < level) continue;

                int64_t dist = computeUpperLayerDist(neighbor);
                if (dist < currDist) {
                    currDist = dist;
                    currNode = neighbor;
                    changed = true;
                }
            }
        }
    }

    return currNode;
}

std::vector<std::pair<int, int64_t>> PrivateHNSWClientV2::search(
    const uint8_t* query, int k, int ef,
    PrivateHNSWServerV2& server
) {
    PrivateSearchStatsV2 stats;
    return searchWithStats(query, k, ef, server, stats);
}

std::vector<std::pair<int, int64_t>> PrivateHNSWClientV2::searchWithStats(
    const uint8_t* query, int k, int ef,
    PrivateHNSWServerV2& server,
    PrivateSearchStatsV2& stats
) {
    clearCache();
    clearSubgroupCiphertextCache();

    auto precomputeStart = std::chrono::high_resolution_clock::now();
    int estimatedClusters = estimatePoolSize(ef);
    precomputeSecretPool(estimatedClusters);
    auto precomputeEnd = std::chrono::high_resolution_clock::now();
    stats.precomputeHsTimeMs = std::chrono::duration<double, std::milli>(precomputeEnd - precomputeStart).count();

    auto searchStart = std::chrono::high_resolution_clock::now();

    int entryCluster = -1;
    std::vector<int> entryClusters;  // 质心入口: top-K 聚类

    if (config_.useCentroidEntry) {
        // 方案B: 质心入口 — 客户端本地遍历所有质心，零通信
        auto upperStart = std::chrono::high_resolution_clock::now();
        bool signedQ = !config_.isUnsigned;
        int K = std::max(1, config_.centroidEntryK);

        // 计算所有质心距离
        std::vector<std::pair<int64_t, int>> centroidDists;
        centroidDists.reserve(metadata_.numClusters);
        for (int c = 0; c < metadata_.numClusters; ++c) {
            int64_t ip = 0;
            const auto& centroid = metadata_.clusterCentroids[c];
            if (signedQ) {
                for (int i = 0; i < metadata_.embeddingDim; ++i)
                    ip += (int64_t)static_cast<int8_t>(query[i]) * (int64_t)static_cast<int8_t>(centroid[i]);
            } else {
                for (int i = 0; i < metadata_.embeddingDim; ++i)
                    ip += (int64_t)query[i] * (int64_t)centroid[i];
            }
            centroidDists.push_back({-ip, c});
        }

        // 取 top-K 最近质心
        int actualK = std::min(K, (int)centroidDists.size());
        std::partial_sort(centroidDists.begin(), centroidDists.begin() + actualK, centroidDists.end());
        for (int i = 0; i < actualK; ++i)
            entryClusters.push_back(centroidDists[i].second);
        entryCluster = entryClusters[0];

        auto upperEnd = std::chrono::high_resolution_clock::now();
        stats.upperLayerSearchTimeMs = std::chrono::duration<double, std::milli>(upperEnd - upperStart).count();
    } else {
        // 方案A: 原始上层贪心搜索
        auto upperStart = std::chrono::high_resolution_clock::now();
        int entryNode = privateGreedySearch(query, metadata_.entryPoint, 0);
        entryCluster = metadata_.nodeToCluster[entryNode];
        entryClusters.push_back(entryCluster);
        auto upperEnd = std::chrono::high_resolution_clock::now();
        stats.upperLayerSearchTimeMs = std::chrono::duration<double, std::milli>(upperEnd - upperStart).count();
    }

    using Candidate = std::pair<int64_t, int>;
    using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;

    MinHeap candidates;
    std::priority_queue<Candidate> results;
    std::vector<bool> visited(metadata_.numNodes, false);  // 共享 visited

    // 聚类访问上限: 防止个别query暴走
    const int maxClusters = metadata_.numClusters / 5;
    const int topCand = config_.topCand;
    int clustersQueried = 0;
    int clustersSkipped = 0;

    // 自适应质心剪枝（负距离空间: factor越大=剪枝越强）
    // 大聚类质心偏差大，需稍宽松（factor略小）
    // signed 数据质心内积分布更紧凑，需更宽松剪枝
    int avgClusterSize = metadata_.numNodes / std::max(1, metadata_.numClusters);
    double centroidPruneFactor;
    if (config_.isUnsigned) {
        if (avgClusterSize > 20000) {
            centroidPruneFactor = 0.78;   // SIFT10M: ~35k/cluster
        } else if (avgClusterSize > 5000) {
            centroidPruneFactor = 0.82;   // SIFT1M: ~11k/cluster
        } else {
            centroidPruneFactor = 0.85;
        }
    } else {
        if (avgClusterSize > 20000) {
            centroidPruneFactor = 0.65;   // signed 大规模
        } else if (avgClusterSize > 5000) {
            centroidPruneFactor = 0.50;   // signed 中规模 (Deep1M)
        } else {
            centroidPruneFactor = 0.80;   // signed 小规模
        }
    }

    // 全局最优质心距离（跨轮累积，用于剪枝远聚类）
    int64_t globalBestCentDist = 0;

    auto addNeighborClusters = [&](int newId) {
        if (!config_.useSubgroupNeighborPIR) return;
        if (newId < 0 || newId >= metadata_.numNodes) return;

        // 从 newId 反推 cluster
        auto it = std::upper_bound(metadata_.clusterOffset.begin(), metadata_.clusterOffset.end(), newId);
        int cluster = (int)(it - metadata_.clusterOffset.begin()) - 1;
        if (cluster < 0 || cluster >= config_.numClusters) return;

        int localIdx = newId - metadata_.clusterOffset[cluster];
        int sgCol = cluster * config_.maxSGPerCluster + localIdx / config_.targetSubgroupSize;
        int localInSG = localIdx % config_.targetSubgroupSize;

        // 仅使用已缓存的子组解密邻居，不发起额外 PIR 查询
        if (!subgroupCiphertextCache_.count(sgCol)) return;
        decryptNodeNeighbors(newId, sgCol, localInSG, stats);
    };

    // === 入口聚类初始化（支持 batch 多聚类） ===
    if ((int)entryClusters.size() > 1) {
        batchQueryClusters(entryClusters, query, server, stats);
    } else {
        queryCluster(entryCluster, query, server, stats);
    }
    clustersQueried += (int)entryClusters.size();

    for (int ki = 0; ki < (int)entryClusters.size(); ++ki) {
        int ec = entryClusters[ki];
        int csStart = metadata_.clusterOffset[ec];
        int csEnd = metadata_.clusterOffset[ec + 1];
        std::vector<Candidate> clusterBest;
        for (int newId = csStart; newId < csEnd; ++newId) {
            int64_t dist = distanceCache_[newId];
            if (dist == INT64_MAX) continue;
            clusterBest.push_back({dist, newId});
        }
        if ((int)clusterBest.size() > config_.maxResultsPerCluster) {
            std::partial_sort(clusterBest.begin(), clusterBest.begin() + config_.maxResultsPerCluster, clusterBest.end());
            clusterBest.resize(config_.maxResultsPerCluster);
        }
        int count = 0;
        for (auto& [dist, newId] : clusterBest) {
            // 入口聚类始终加入 top-1 作为种子，不受 topCand=0 影响
            if (count < std::max(topCand, 1)) {
                visited[newId] = true;
                candidates.push({dist, newId});
            }
            results.push({dist, newId});
            stats.resultsFromCluster++;
            if ((int)results.size() > ef) results.pop();
            count++;
        }
    }

    // === HNSW 主循环 ===
    const int d = metadata_.embeddingDim;

    bool signedQuant = !config_.isUnsigned;
    auto computeCentroidDist = [&](int cid) -> int64_t {
        int64_t ip = 0;
        const auto& centroid = metadata_.clusterCentroids[cid];
        if (signedQuant) {
            // signed 数据: uint8 位模式还原为 int8 后计算内积
            for (int i = 0; i < d; ++i)
                ip += (int64_t)static_cast<int8_t>(query[i]) * (int64_t)static_cast<int8_t>(centroid[i]);
        } else {
            for (int i = 0; i < d; ++i)
                ip += (int64_t)query[i] * (int64_t)centroid[i];
        }
        return -ip;
    };

    // 用入口聚类初始化全局剪枝基准
    globalBestCentDist = computeCentroidDist(entryCluster);

    // 跨轮待访问聚类队列（按质心距离排序，小=近=优先）
    std::set<std::pair<int64_t, int>> pendingClusters;  // (centDist, cid)
    // no_prune 模式：不排序不剪枝，按发现顺序（FIFO）取聚类
    std::vector<int> pendingClustersFIFO;
    std::set<int> pendingClustersFIFOSet;  // 去重用

    // dummy ID 用于填充（入口聚类已缓存，查询结果会被忽略）
    const int dummyCluster = entryClusters[0];
    const int totalSubgroups = config_.numClusters * config_.maxSGPerCluster;

    // 选取一个未缓存的随机子组作为 dummy，避免被 batchQuerySubgroups 内部过滤
    auto pickUncachedSubgroup = [&]() -> int {
        // 从随机位置开始线性扫描，找到未缓存的子组
        int start = rand() % totalSubgroups;
        for (int i = 0; i < totalSubgroups; i++) {
            int col = (start + i) % totalSubgroups;
            if (!subgroupCiphertextCache_.count(col)) return col;
        }
        return 0;  // 兜底（理论上不会发生）
    };

    // === 固定迭代主循环: 每次 1 NbrPIR + 1 EmbPIR ===
    for (int iter = 0; iter < config_.fixedSearchIterations; iter++) {

        // === Phase 1+2: 弹出候选 + 收集未缓存子组（固定 NbrPIR batch 大小） ===
        std::vector<Candidate> candidateBatch;
        std::set<int> neededSubgroupSet;  // 去重
        {
            while (!candidates.empty()) {
                auto [dist, node] = candidates.top();
                if ((int)results.size() >= ef && dist > results.top().first) break;
                // 计算子组
                auto cit = std::upper_bound(metadata_.clusterOffset.begin(), metadata_.clusterOffset.end(), node);
                int cluster = (int)(cit - metadata_.clusterOffset.begin()) - 1;
                int localIdx = node - metadata_.clusterOffset[cluster];
                int sgCol = cluster * config_.maxSGPerCluster + localIdx / config_.targetSubgroupSize;
                bool cached = subgroupCiphertextCache_.count(sgCol) || neededSubgroupSet.count(sgCol);
                // 未缓存子组已满，只接受缓存命中的候选
                if (!cached && (int)neededSubgroupSet.size() >= config_.maxCandidatesPerRound) break;
                candidates.pop();
                candidateBatch.push_back({dist, node});
                if (!cached) neededSubgroupSet.insert(sgCol);
            }
        }
        // 构建 NbrPIR batch（固定大小，用未缓存的随机子组填充）
        std::vector<int> neededSubgroups(neededSubgroupSet.begin(), neededSubgroupSet.end());
        while ((int)neededSubgroups.size() < config_.maxCandidatesPerRound)
            neededSubgroups.push_back(pickUncachedSubgroup());
        neededSubgroups.resize(config_.maxCandidatesPerRound);
        batchQuerySubgroups(neededSubgroups, server, stats);

        // === Phase 3: 逐个处理候选（NbrPIR 已缓存）===
        std::unordered_map<int, std::vector<int>> allUncachedClusterNeighbors;
        for (int idx = 0; idx < (int)candidateBatch.size(); ++idx) {
            auto [dist, currNewId] = candidateBatch[idx];
            addNeighborClusters(currNewId);
            auto it = neighborNodeIds_.find(currNewId);
            if (it == neighborNodeIds_.end()) continue;
            for (int nbrNewId : it->second) {
                if (visited[nbrNewId]) continue;
                auto cit = std::upper_bound(metadata_.clusterOffset.begin(), metadata_.clusterOffset.end(), nbrNewId);
                int nbrCluster = (int)(cit - metadata_.clusterOffset.begin()) - 1;
                if (isClusterCached(nbrCluster)) {
                    visited[nbrNewId] = true;
                    int64_t d2 = distanceCache_[nbrNewId];
                    if (d2 != INT64_MAX) {
                        stats.nodesVisited++;
                        if ((int)results.size() < ef || d2 < results.top().first) {
                            candidates.push({d2, nbrNewId});
                            results.push({d2, nbrNewId});
                            stats.resultsFromNeighbor++;
                            if ((int)results.size() > ef) results.pop();
                        }
                    }
                } else {
                    allUncachedClusterNeighbors[nbrCluster].push_back(nbrNewId);
                }
            }
        }

        // === Phase 4: EmbPIR 批量查询（固定 batch 大小） ===
        // 将本轮新发现的未缓存聚类加入 pendingClusters
        for (auto& [cid, nbrIds] : allUncachedClusterNeighbors) {
            if (isClusterCached(cid)) continue;
            if (config_.disableCentroidPrune) {
                // no_prune 模式：不计算质心距离，按发现顺序加入
                if (!pendingClustersFIFOSet.count(cid)) {
                    pendingClustersFIFO.push_back(cid);
                    pendingClustersFIFOSet.insert(cid);
                }
            } else {
                int64_t centDist = computeCentroidDist(cid);
                if (globalBestCentDist == 0 || centDist < globalBestCentDist)
                    globalBestCentDist = centDist;
                pendingClusters.insert({centDist, cid});
            }
        }
        // 质心排序+剪枝（仅非 no_prune 模式）
        if (!config_.disableCentroidPrune && globalBestCentDist < 0) {
            int64_t pruneThreshold = static_cast<int64_t>(globalBestCentDist * centroidPruneFactor);
            auto pit = pendingClusters.begin();
            while (pit != pendingClusters.end()) {
                if (pit->first > pruneThreshold) {
                    clustersSkipped++;
                    pit = pendingClusters.erase(pit);
                } else {
                    ++pit;
                }
            }
        }
        // 从 pendingClusters 取真实聚类
        std::vector<int> batchClusterIds;
        if (config_.disableCentroidPrune) {
            // no_prune 模式：按发现顺序（FIFO）取聚类
            auto it = pendingClustersFIFO.begin();
            while (it != pendingClustersFIFO.end() && (int)batchClusterIds.size() < config_.maxClustersPerRound) {
                if (isClusterCached(*it)) {
                    pendingClustersFIFOSet.erase(*it);
                    it = pendingClustersFIFO.erase(it);
                    continue;
                }
                batchClusterIds.push_back(*it);
                pendingClustersFIFOSet.erase(*it);
                it = pendingClustersFIFO.erase(it);
            }
        } else {
            auto pit = pendingClusters.begin();
            while (pit != pendingClusters.end() && (int)batchClusterIds.size() < config_.maxClustersPerRound) {
                if (isClusterCached(pit->second)) {
                    pit = pendingClusters.erase(pit);
                    continue;
                }
                batchClusterIds.push_back(pit->second);
                pit = pendingClusters.erase(pit);
            }
        }
        int realClusterCount = (int)batchClusterIds.size();
        // 补齐到固定大小（用已缓存的入口聚类填充，不影响搜索结果）
        while ((int)batchClusterIds.size() < config_.maxClustersPerRound)
            batchClusterIds.push_back(dummyCluster);  // EmbPIR 无内部过滤，dummy 不会被跳过
        batchQueryClusters(batchClusterIds, query, server, stats);
        clustersQueried += realClusterCount;

        // 处理 EmbPIR 结果（仅真实聚类）
        for (int i = 0; i < realClusterCount; i++) {
            int cid = batchClusterIds[i];
            // 处理 uncached 邻居
            for (int nbrNewId : allUncachedClusterNeighbors[cid]) {
                if (visited[nbrNewId]) continue;
                visited[nbrNewId] = true;
                int64_t d2 = distanceCache_[nbrNewId];
                if (d2 == INT64_MAX) continue;
                stats.nodesVisited++;
                if ((int)results.size() < ef || d2 < results.top().first) {
                    candidates.push({d2, nbrNewId});
                    results.push({d2, nbrNewId});
                    stats.resultsFromNeighbor++;
                    if ((int)results.size() > ef) results.pop();
                }
            }
            // topCand 逻辑
            int csStart = metadata_.clusterOffset[cid];
            int csEnd = metadata_.clusterOffset[cid + 1];
            std::vector<Candidate> clusterBest;
            for (int newId = csStart; newId < csEnd; ++newId) {
                if (visited[newId]) continue;
                int64_t d2 = distanceCache_[newId];
                if (d2 == INT64_MAX) continue;
                clusterBest.push_back({d2, newId});
            }
            if ((int)clusterBest.size() > config_.maxResultsPerCluster) {
                std::partial_sort(clusterBest.begin(), clusterBest.begin() + config_.maxResultsPerCluster, clusterBest.end());
                clusterBest.resize(config_.maxResultsPerCluster);
            }
            int count = 0;
            for (auto& [d2, newId] : clusterBest) {
                stats.nodesVisited++;
                if (topCand > 0 && count < topCand) {
                    visited[newId] = true;
                    candidates.push({d2, newId});
                }
                if ((int)results.size() < ef || d2 < results.top().first) {
                    results.push({d2, newId});
                    stats.resultsFromCluster++;
                    if ((int)results.size() > ef) results.pop();
                }
                count++;
            }
        }
    }

    // Extract top-k results（newId → 原始 nodeId）
    std::vector<std::pair<int, int64_t>> topK;
    while (!results.empty()) {
        int newId = results.top().second;
        topK.push_back({metadata_.newIdToNode[newId], results.top().first});
        results.pop();
    }
    std::reverse(topK.begin(), topK.end());
    if ((int)topK.size() > k) {
        topK.resize(k);
    }

    auto searchEnd = std::chrono::high_resolution_clock::now();
    stats.totalSearchTimeMs = std::chrono::duration<double, std::milli>(searchEnd - searchStart).count();
    stats.clustersAccessed = std::count(cachedClusters_.begin(), cachedClusters_.end(), true);
    stats.clustersSkipped = clustersSkipped;

    lastEmbPoolUsed_ = embPoolIdx_;
    lastNbrPoolUsed_ = nbrPoolIdx_;
    queryCount_++;
    const double alpha = 0.3;
    if (queryCount_ == 1) {
        avgEmbPoolUsage_ = embPoolIdx_;
        avgNbrPoolUsage_ = nbrPoolIdx_;
    } else {
        avgEmbPoolUsage_ = alpha * embPoolIdx_ + (1 - alpha) * avgEmbPoolUsage_;
        avgNbrPoolUsage_ = alpha * nbrPoolIdx_ + (1 - alpha) * avgNbrPoolUsage_;
    }

    return topK;
}

// ============================================================================
// PrivateHNSWUtilsV2 — 命名规范化
// ============================================================================

std::shared_ptr<EmbMatrix> PrivateHNSWUtilsV2::generateEmbeddingSharedMatrix(
    const simplepir::EmbeddingPIRParams& params
) {
    uint64_t rows = params.config.queryDim();
    uint64_t cols = params.pirParams.N;
    return EmbMatrix::random(rows, cols, params.pirParams.Logq);
}

std::shared_ptr<NbrMatrix> PrivateHNSWUtilsV2::generateNeighborSharedMatrix(
    const simplepir::NeighborPIRParams& params
) {
    return NbrMatrix::random(params.pirParams.M, params.pirParams.N, params.pirParams.Logq);
}



// ============================================
// 辅助函数
// ============================================


// ============================================
// 子组密文缓存相关函数
// ============================================

void PrivateHNSWClientV2::clearSubgroupCiphertextCache() {
    subgroupCiphertextCache_.clear();
}

void PrivateHNSWClientV2::querySubgroupCiphertext(
    int subgroupColumn,
    PrivateHNSWServerV2& server,
    PrivateSearchStatsV2& stats
) {
    if (subgroupCiphertextCache_.count(subgroupColumn)) {
        return;
    }

    int M0 = metadata_.maxNeighbors;
    int maxSubgroupSize = config_.maxSubgroupSize;
    int numParts = config_.numParts;
    int totalRows = maxSubgroupSize * M0 * numParts;

    // 1. 使用 NeighborPIRClient 生成查询
    auto nbrQueryGenStart = std::chrono::high_resolution_clock::now();

    auto nbrPrecompute = getNbrPrecompute();
    auto [queryMsg, ctx] = nbrPirClient_.query(
        subgroupColumn, nbrPrecompute.As, nbrPrecompute.Hs);

    auto nbrQueryGenEnd = std::chrono::high_resolution_clock::now();
    stats.nbrQueryGenTimeMs += std::chrono::duration<double, std::milli>(nbrQueryGenEnd - nbrQueryGenStart).count();

    // 2. 服务端计算
    auto nbrServerStart = std::chrono::high_resolution_clock::now();
    NbrAnswerMsg nbrAnswer = server.answerNeighborPIR(queryMsg);
    auto nbrServerEnd = std::chrono::high_resolution_clock::now();
    stats.nbrServerTimeMs += std::chrono::duration<double, std::milli>(nbrServerEnd - nbrServerStart).count();

    stats.neighborPirCount++;
    stats.nbrCommRounds += 1;
    stats.totalPirTimeMs += std::chrono::duration<double, std::milli>(nbrServerEnd - nbrQueryGenStart).count();

    // 3. 缓存密文和 Hs
    SubgroupCiphertextCache cache;
    cache.answer = nbrAnswer.answer;
    cache.Hs = ctx.Hs;
    cache.subgroupSize = maxSubgroupSize;
    cache.decryptedNodes.resize(maxSubgroupSize, false);
    subgroupCiphertextCache_[subgroupColumn] = std::move(cache);

    // 通信开销统计
    uint64_t nbrElemSize = sizeof(NbrElem);
    stats.nbrQueryBytes += nbrPirParams_.pirParams.M * nbrElemSize;
    stats.nbrAnswerBytes += totalRows * nbrElemSize;
}

void PrivateHNSWClientV2::batchQuerySubgroups(
    const std::vector<int>& subgroupColumns,
    PrivateHNSWServerV2& server,
    PrivateSearchStatsV2& stats
) {
    // 过滤已缓存的子组
    std::vector<int> toQuery;
    for (int col : subgroupColumns) {
        if (!subgroupCiphertextCache_.count(col)) {
            toQuery.push_back(col);
        }
    }
    if (toQuery.empty()) return;
    // 消融: 关闭批量 PIR 时逐个查询
    if (config_.disableBatchPIR || toQuery.size() == 1) {
        for (int col : toQuery) {
            querySubgroupCiphertext(col, server, stats);
        }
        return;
    }

    int N = (int)toQuery.size();
    uint64_t M = nbrPirParams_.pirParams.M;
    uint64_t L = nbrPirParams_.pirParams.L;
    uint64_t delta = nbrPirParams_.delta();

    auto nbrQueryGenStart = std::chrono::high_resolution_clock::now();

    // 构建 M × N 批量查询矩阵
    auto batchQuery = std::make_shared<NbrMatrix>(M, N);
    std::vector<std::shared_ptr<NbrMatrix>> hsVec(N);

    // 一次生成所有噪声 (M × N)
    auto noise = NbrMatrix::gaussian(M, N);
    const NbrElem* noiseData = noise->data.data();
    NbrElem* bqData = batchQuery->data.data();

    for (int j = 0; j < N; j++) {
        auto pre = getNbrPrecompute();
        hsVec[j] = pre.Hs;

        const NbrElem* asData = pre.As->data.data();

        // 合并 As 拷贝 + 噪声（直接操作 data 数组，跳过边界检查）
        for (uint64_t r = 0; r < M; r++) {
            bqData[r * N + j] = asData[r] + noiseData[r * N + j];
        }

        // delta·e_{toQuery[j]}
        bqData[toQuery[j] * N + j] += static_cast<NbrElem>(delta);
    }

    auto nbrQueryGenEnd = std::chrono::high_resolution_clock::now();
    stats.nbrQueryGenTimeMs += std::chrono::duration<double, std::milli>(nbrQueryGenEnd - nbrQueryGenStart).count();

    // 服务端批量计算
    auto nbrServerStart = std::chrono::high_resolution_clock::now();
    auto batchAns = server.batchAnswerNeighborPIR(batchQuery);
    auto nbrServerEnd = std::chrono::high_resolution_clock::now();
    stats.nbrServerTimeMs += std::chrono::duration<double, std::milli>(nbrServerEnd - nbrServerStart).count();

    // 逐列缓存（直接操作 data 数组，跳过边界检查）
    const NbrElem* ansData = batchAns->data.data();
    for (int j = 0; j < N; j++) {
        int col = toQuery[j];
        SubgroupCiphertextCache cache;
        auto colAns = std::make_shared<NbrMatrix>(L, 1);
        NbrElem* dst = colAns->data.data();
        for (uint64_t r = 0; r < L; r++)
            dst[r] = ansData[r * N + j];
        cache.answer = colAns;
        cache.Hs = hsVec[j];
        cache.subgroupSize = config_.maxSubgroupSize;
        cache.decryptedNodes.resize(config_.maxSubgroupSize, false);
        subgroupCiphertextCache_[col] = std::move(cache);
    }

    stats.neighborPirCount += N;
    stats.nbrCommRounds += 1;
    stats.totalPirTimeMs += std::chrono::duration<double, std::milli>(nbrServerEnd - nbrQueryGenStart).count();

    uint64_t nbrElemSize = sizeof(NbrElem);
    stats.nbrQueryBytes += M * N * nbrElemSize;
    stats.nbrAnswerBytes += L * N * nbrElemSize;
}


void PrivateHNSWClientV2::decryptNodeNeighbors(
    int newId,
    int subgroupColumn,
    int localIdx,
    PrivateSearchStatsV2& stats
) {
    auto it = subgroupCiphertextCache_.find(subgroupColumn);
    if (it == subgroupCiphertextCache_.end()) {
        return;
    }

    auto& cache = it->second;

    if (localIdx >= 0 && localIdx < (int)cache.decryptedNodes.size() && cache.decryptedNodes[localIdx]) {
        return;
    }

    auto nbrRecoverStart = std::chrono::high_resolution_clock::now();

    int M0 = metadata_.maxNeighbors;
    int numParts = config_.numParts;
    int partBits = config_.partBits;
    int localBits = config_.localBits;
    uint32_t partMask = (1u << partBits) - 1;
    uint32_t INVALID_PART = static_cast<uint32_t>((1u << partBits) - 1);  // P - 1 = 127

    // 使用 NeighborPIRClient 恢复 M0 * numParts 个原始值
    NbrAnswerMsg ansMsg;
    ansMsg.answer = cache.answer;
    NbrQueryContext ctx(cache.Hs);
    auto recoveredValues = nbrPirClient_.recoverNodeValues(ansMsg, ctx, localIdx);

    // 确保 neighborCache_[newId] 已初始化
    if (newId < 0 || newId >= (int)neighborCache_.size()) {
        std::cerr << "[decryptNodeNeighbors] ERROR: newId=" << newId
                  << " out of bounds (size=" << neighborCache_.size() << ")\n";
        return;
    }
    if (neighborCache_[newId].empty()) {
        neighborCache_[newId].resize(metadata_.numClusters, false);
    }

    // 解码: 重组 numParts 个部分为完整编码，O(1) 解码
    for (int j = 0; j < M0; ++j) {
        // 重组 numParts 个部分
        uint32_t encoded = 0;
        bool isInvalid = true;
        for (int p = 0; p < numParts; ++p) {
            uint64_t part = recoveredValues[j * numParts + p];
            if (part != INVALID_PART) isInvalid = false;
            encoded |= (static_cast<uint32_t>(part) & partMask) << (p * partBits);
        }
        if (isInvalid) continue;

        // O(1) 解码: (clusterIdx << localBits) | localIdxInCluster
        int nbrLocalInCluster = encoded & ((1u << localBits) - 1);
        int nbrCluster = encoded >> localBits;

        if (nbrCluster < 0 || nbrCluster >= metadata_.numClusters) continue;
        if (nbrCluster >= (int)metadata_.clusterOffset.size() - 1) continue;

        int nbrNewId = metadata_.clusterOffset[nbrCluster] + nbrLocalInCluster;
        if (nbrNewId < 0 || nbrNewId >= (int)metadata_.newIdToNode.size()) continue;

        // 直接存储 nbrNewId（不再转回 nodeId）
        neighborNodeIds_[newId].push_back(nbrNewId);
        neighborCache_[newId][nbrCluster] = true;
    }

    // 标记已解密
    if (localIdx >= 0 && localIdx < (int)cache.decryptedNodes.size()) {
        cache.decryptedNodes[localIdx] = true;
    }

    auto nbrRecoverEnd = std::chrono::high_resolution_clock::now();
    stats.nbrRecoverTimeMs += std::chrono::duration<double, std::milli>(nbrRecoverEnd - nbrRecoverStart).count();
}

} // namespace hnsw

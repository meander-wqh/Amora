/**
 * @file private_hnsw_v2.h
 * @brief Privacy-preserving HNSW search with neighbor PIR
 *
 * This version uses two separate PIRs (DUAL mode):
 * - Embedding PIR: returns inner products for a cluster
 * - Neighbor PIR: returns neighbor subgroup information (subgroup-level encoding)
 *
 * 所有 PIR 模块已模板化，通过顶层类型别名切换 32/64 位。
 */

#ifndef PRIVATE_HNSW_V2_H
#define PRIVATE_HNSW_V2_H

#include "hnsw_quantized.h"
#include "subgroup/subgroup_manager.h"
#include "../simplepir/embedding_pir.h"
#include "../simplepir/neighbor_pir.h"
#include "../simplepir/simple_pir.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>
#include <chrono>
#include <cmath>

namespace hnsw {

// ============================================
// PIR 元素类型配置（修改此处即可全局切换 32/64 位）
// ============================================
using EmbElem = simplepir::Elem32;
using NbrElem = simplepir::Elem32;

// EmbeddingPIR 类型别名
using EmbMatrix = simplepir::MatrixT<EmbElem>;
using EmbQueryMsg = simplepir::QueryMsgT<EmbElem>;
using EmbAnswerMsg = simplepir::AnswerMsgT<EmbElem>;
using EmbQueryContext = simplepir::QueryContextT<EmbElem>;
using EmbDatabase = simplepir::EmbeddingDatabaseT<EmbElem>;
using EmbPIRServer = simplepir::EmbeddingPIRServerT<EmbElem>;
using EmbPIRClient = simplepir::EmbeddingPIRClientT<EmbElem>;

// NeighborPIR 类型别名
using NbrMatrix = simplepir::MatrixT<NbrElem>;
using NbrQueryMsg = simplepir::NbrQueryMsgT<NbrElem>;
using NbrAnswerMsg = simplepir::NbrAnswerMsgT<NbrElem>;
using NbrQueryContext = simplepir::NbrQueryContextT<NbrElem>;
using NbrDatabase = simplepir::NeighborDatabaseT<NbrElem>;
using NbrPIRServer = simplepir::NeighborPIRServerT<NbrElem>;
using NbrPIRClient = simplepir::NeighborPIRClientT<NbrElem>;

// 统一预计算结构
template<typename ElemType>
struct PIRPrecomputeT {
    std::shared_ptr<simplepir::MatrixT<ElemType>> As;
    std::shared_ptr<simplepir::MatrixT<ElemType>> Hs;
};
using EmbPIRPrecompute = PIRPrecomputeT<EmbElem>;
using NbrPIRPrecompute = PIRPrecomputeT<NbrElem>;

/**
 * @brief Configuration for Private HNSW V2
 */
struct PrivateHNSWConfigV2 {
    int embeddingDim = 0;
    int numNodes = 0;
    int numClusters = 0;
    int maxClusterSize = 0;
    int maxNeighbors = 0;      // M0 = 2*M
    uint64_t logQ = 32;              // Neighbor PIR logQ (32-bit for Elem32)
    uint64_t embeddingLogQ = 32;     // Embedding PIR logQ (32-bit for Elem32)
    uint64_t lweN = 1024;
    bool isUnsigned = true;          // 量化模式：true=unsigned[0,255], false=signed[-128,127]

    // ============================================
    // 子组级 NeighborPIR 配置
    // ============================================
    bool useSubgroupNeighborPIR = false;  // 是否使用子组级 NeighborPIR
    int totalSubgroups = 0;               // 总子组数
    int maxSubgroupSize = 0;              // 子组最大节点数
    int targetSubgroupSize = 120;         // 目标子组大小
    int maxSGPerCluster = 0;              // ceil(maxClusterSize / targetSubgroupSize)

    // 拆分编码参数
    int partBits = 7;                     // 每部分的位数
    int numParts = 3;                     // 编码拆分的部分数
    int localBits = 14;                   // localIdxInCluster 的位数
    int clusterBits = 7;                  // clusterIdx 的位数

    // ============================================
    // 消融实验开关
    // ============================================
    bool disableCentroidPrune = false;    // 关闭质心剪枝（所有候选聚类都查询）
    int topCand = 1;                      // 每聚类取几个候选加入候选集（0=不加入）
    bool disableBatchPIR = false;         // 关闭批量 PIR（每个聚类/子组单独查询）
    int maxEmbRounds = 0;                 // EmbPIR 最大轮次（0=不限制）
    bool useCentroidEntry = true;         // 用最近质心聚类作入口（替代上层贪心搜索）
    int centroidEntryK = 1;               // 质心入口时 batch 查询的聚类数（与 NbrPIR batch 对齐）, msmarco = 3, sift10m = 1, deep1m = 1
    int maxStaleRounds = 0;               // 连续无改善轮次上限（0=不限制）
    int maxCandidatesPerRound = 2;        // 每轮最多弹出的候选节点数（=NbrPIR batch大小）, msmarco = 3, sift10m = 2, deep1m = 2
    int maxClustersPerRound = 2;          // 每轮 EmbPIR 最大聚类数（=EmbPIR batch大小）, msmarco = 4, sift10m = 2, deep1m = 3
    int fixedSearchIterations = 5;        // 固定主循环迭代次数（每次 1 NbrPIR + 1 EmbPIR）
    int maxResultsPerCluster = 20;        // 每聚类加入 result 的最大节点数
    void print() const;
};

/**
 * @brief Minimal metadata for client (no neighbor info exposed)
 */
struct PrivateHNSWMetadataV2 {
    int numNodes = 0;
    int numClusters = 0;
    int maxClusterSize = 0;
    int embeddingDim = 0;
    int maxNeighbors = 0;

    // Entry point info
    int entryPoint = 0;
    int entryCluster = 0;

    // Cluster membership (needed to interpret PIR results)
    std::vector<std::vector<int>> clusterToNodes;  // cluster -> node IDs
    std::vector<int> nodeToCluster;                // node -> cluster
    std::vector<int> nodeIndexInCluster;           // node -> index within cluster

    // Upper layer info (for greedy search without PIR)
    int maxLevel = 0;
    std::vector<int> nodeLevels;
    std::vector<std::vector<std::vector<int>>> allLevelNeighbors;  // node -> level -> neighbors
    std::vector<int> upperLayerNodes;
    std::vector<std::vector<uint8_t>> upperLayerEmbeddings;
    std::vector<int> nodeToUpperIdx;

    // ============================================
    // 聚类质心 (用于聚类级别剪枝)
    // ============================================
    std::vector<std::vector<uint8_t>> clusterCentroids;  // cluster -> centroid embedding (quantized)

    // ============================================
    // 子组信息 (用于选择性解密)
    // ============================================
    bool hasSubgrouping = false;
    int totalSubgroups = 0;
    int maxSubgroupSize = 0;

    // 子组边界: clusterSubgroupOffset[c][g] = 聚类c子组g的起始节点索引（相对于聚类）
    std::vector<std::vector<int>> clusterSubgroupOffset;

    // 节点ID映射: nodeToNewId[oldId] = newId (按聚类+子组顺序)
    std::vector<int> nodeToNewId;
    std::vector<int> newIdToNode;

    // 聚类偏移: clusterOffset[c] = 聚类c的起始newId
    std::vector<int> clusterOffset;

    // 直接子组映射: nodeSubgroup[newId] -> subgroupId (聚类内子组索引)
    std::vector<int> nodeSubgroup;

    // 直接映射: nodeLocalIdxInSubgroup[newId] -> 节点在子组内的局部索引
    std::vector<int> nodeLocalIdxInSubgroup;

    // 邻居子组信息（直接使用 hnsw::NodeNeighborInfo，避免类型转换拷贝）
    std::vector<NodeNeighborInfo> nodeNeighborGroups;  // node -> NodeNeighborInfo

    // ============================================
    // 子组级 NeighborPIR 映射 (新方案)
    // ============================================
    std::vector<std::vector<int>> subgroupToColumn;
    std::vector<int> numSubgroupsPerCluster;

    // 辅助函数
    int getLocalIndexInSubgroup(int nodeId) const;
    std::pair<int, int> getNodeSubgroup(int nodeId) const;

    void print() const;
};

/**
 * @brief Result from a cluster PIR query
 */
struct ClusterPIRResult {
    std::vector<int64_t> innerProducts;              // K inner products
    std::vector<std::vector<bool>> neighborClusters; // K × C bitmap
};

/**
 * @brief Search statistics
 */
struct PrivateSearchStatsV2 {
    int pirQueryCount = 0;
    int embeddingPirCount = 0;
    int neighborPirCount = 0;
    int clustersAccessed = 0;
    int clustersSkipped = 0;
    int nodesVisited = 0;
    int resultsFromNeighbor = 0;   // 通过邻居路径加入 results 的节点数
    int resultsFromCluster = 0;    // 通过聚类扫描加入 results 的节点数
    int cacheHits = 0;
    double totalPirTimeMs = 0;
    double totalSearchTimeMs = 0;

    // Communication costs (bytes) for a single query
    uint64_t embQueryBytes = 0;
    uint64_t embAnswerBytes = 0;
    uint64_t nbrQueryBytes = 0;
    uint64_t nbrAnswerBytes = 0;

    uint64_t totalQueryBytes() const { return embQueryBytes + nbrQueryBytes; }
    uint64_t totalAnswerBytes() const { return embAnswerBytes + nbrAnswerBytes; }
    uint64_t totalBytes() const { return totalQueryBytes() + totalAnswerBytes(); }

    // 通信轮次统计
    int embCommRounds = 0;    // EmbPIR 通信轮次（1次batch = 1轮）
    int nbrCommRounds = 0;    // NbrPIR 通信轮次（每次查询 = 1轮）
    int totalCommRounds() const { return embCommRounds + nbrCommRounds; }

    // ============================================
    // 详细计时 (用于性能分析)
    // ============================================
    double precomputeHsTimeMs = 0;
    double upperLayerSearchTimeMs = 0;
    double candidateManageTimeMs = 0;

    // Embedding PIR 细分
    double embQueryGenTimeMs = 0;
    double embServerTimeMs = 0;
    double embRecoverTimeMs = 0;

    // Neighbor PIR 细分
    double nbrQueryGenTimeMs = 0;
    double nbrServerTimeMs = 0;
    double nbrRecoverTimeMs = 0;
    double nbrDecodeTimeMs = 0;

    // 缓存操作
    double cacheWriteTimeMs = 0;

    // 汇总
    double totalEmbPirTimeMs() const { return embQueryGenTimeMs + embServerTimeMs + embRecoverTimeMs; }
    double totalNbrPirTimeMs() const { return nbrQueryGenTimeMs + nbrServerTimeMs + nbrRecoverTimeMs + nbrDecodeTimeMs; }
    double onlineTimeMs() const {
        return totalEmbPirTimeMs() + totalNbrPirTimeMs() + upperLayerSearchTimeMs +
               candidateManageTimeMs + cacheWriteTimeMs;
    }

    void printDetailedTiming() const;
};

/**
 * @brief Server for Private HNSW V2
 */
class PrivateHNSWServerV2 {
public:
    PrivateHNSWServerV2() = default;

    /**
     * @brief Build server from quantized index
     */
    void build(HNSWQuantizedIndex& index);

    /**
     * @brief Setup Embedding PIR（对称接口）
     * @param sharedMatrix Shared random matrix A
     * @return Embedding hint matrix
     */
    std::shared_ptr<EmbMatrix> setupEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& sharedMatrix
    );

    /**
     * @brief Setup Embedding PIR（从缓存加载 hint，只压缩 DB）
     */
    void setupEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& sharedMatrix,
        const std::shared_ptr<EmbMatrix>& cachedHint
    );

    /**
     * @brief Setup Neighbor PIR（对称接口）
     * @param sharedMatrix Shared random matrix A
     * @return Neighbor hint matrix
     */
    std::shared_ptr<NbrMatrix> setupNeighborPIR(
        const std::shared_ptr<NbrMatrix>& sharedMatrix
    );

    /**
     * @brief Setup Neighbor PIR（从缓存加载 hint，只压缩 DB）
     */
    void setupNeighborPIR(
        const std::shared_ptr<NbrMatrix>& sharedMatrix,
        const std::shared_ptr<NbrMatrix>& cachedHint
    );

    /**
     * @brief Answer embedding PIR query
     */
    EmbAnswerMsg answerEmbeddingPIR(const EmbQueryMsg& query) const;

    /**
     * @brief Answer neighbor PIR query
     */
    NbrAnswerMsg answerNeighborPIR(const NbrQueryMsg& query) const;

    /**
     * @brief 批量应答 Embedding PIR 查询
     */
    std::shared_ptr<EmbMatrix> batchAnswerEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& queryMatrix) const;

    /**
     * @brief 批量应答 Neighbor PIR 查询
     */
    std::shared_ptr<NbrMatrix> batchAnswerNeighborPIR(
        const std::shared_ptr<NbrMatrix>& queryMatrix) const;

    // Getters
    const PrivateHNSWConfigV2& getConfig() const { return config_; }
    const PrivateHNSWMetadataV2& getMetadata() const { return metadata_; }
    const simplepir::EmbeddingPIRParams& getEmbeddingPIRParams() const { return embPirParams_; }
    const simplepir::NeighborPIRParams& getNeighborPIRParams() const { return nbrPirParams_; }

    bool isReady() const { return isReady_; }

private:
    void buildMetadata(HNSWQuantizedIndex& index);
    void buildSubgroupNeighborPIRDatabase(const HNSWQuantizedIndex& index);

    PrivateHNSWConfigV2 config_;
    PrivateHNSWMetadataV2 metadata_;
    SubgroupManager subgroupManager_;

    // Embedding PIR（对称）
    simplepir::EmbeddingPIRParams embPirParams_;
    EmbDatabase embDatabase_;
    EmbPIRServer embPirServer_;

    // Neighbor PIR（对称）
    simplepir::NeighborPIRParams nbrPirParams_;
    NbrDatabase nbrDatabase_;
    NbrPIRServer nbrPirServer_;

    bool isReady_ = false;
};

/**
 * @brief Client for Private HNSW V2
 */
class PrivateHNSWClientV2 {
public:
    PrivateHNSWClientV2() = default;

    /**
     * @brief 基础初始化（metadata, config, cache）
     */
    void init(
        const PrivateHNSWMetadataV2& metadata,
        const PrivateHNSWConfigV2& config
    );

    /**
     * @brief 对称的 Embedding PIR 初始化
     */
    void initEmbeddingPIR(
        const simplepir::EmbeddingPIRParams& params,
        const std::shared_ptr<EmbMatrix>& sharedMatrix,
        const std::shared_ptr<EmbMatrix>& hint
    );

    /**
     * @brief 对称的 Neighbor PIR 初始化
     */
    void initNeighborPIR(
        const simplepir::NeighborPIRParams& params,
        const std::shared_ptr<NbrMatrix>& sharedMatrix,
        const std::shared_ptr<NbrMatrix>& hint
    );

    /**
     * @brief Perform private search
     */
    std::vector<std::pair<int, int64_t>> search(
        const uint8_t* query, int k, int ef,
        PrivateHNSWServerV2& server
    );

    /**
     * @brief Perform private search with statistics
     */
    std::vector<std::pair<int, int64_t>> searchWithStats(
        const uint8_t* query, int k, int ef,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    void clearCache();
    bool isReady() const { return isReady_; }

    /**
     * @brief 预计算 PIR 查询池（安全优化版本）
     */
    void precomputeSecretPool(int poolSize);

private:
    // 批量查询多个聚类的 EmbeddingPIR（合并为1次矩阵乘法）
    void batchQueryClusters(
        const std::vector<int>& clusterIds,
        const uint8_t* query,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    // Query a cluster and get both inner products and neighbor info
    ClusterPIRResult queryCluster(
        int clusterId,
        const uint8_t* query,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    // Upper layer greedy search (no PIR)
    int privateGreedySearch(
        const uint8_t* query,
        int entryPoint,
        int targetLevel
    );

    // Cache management
    bool isClusterCached(int clusterId) const;
    int64_t getCachedDistance(int newId) const;
    bool hasNeighborInCluster(int newId, int clusterId) const;

    // ============================================
    // 子组密文缓存（按需解密优化）
    // ============================================
    struct SubgroupCiphertextCache {
        std::shared_ptr<NbrMatrix> answer;        // 服务器返回的密文
        std::shared_ptr<NbrMatrix> Hs;            // 预计算的 Hs
        std::vector<bool> decryptedNodes;         // 哪些节点已解密
        int subgroupSize;                         // 子组大小
    };
    std::unordered_map<int, SubgroupCiphertextCache> subgroupCiphertextCache_;

    // 只获取子组密文（不解密）
    void querySubgroupCiphertext(
        int subgroupColumn,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    // NbrPIR 批量查询多个子组（合并为1次矩阵乘法）
    void batchQuerySubgroups(
        const std::vector<int>& subgroupColumns,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    // 按需解密单个节点的邻居信息
    void decryptNodeNeighbors(
        int newId,
        int subgroupColumn,
        int localIdx,
        PrivateSearchStatsV2& stats
    );

    // 清空子组密文缓存
    void clearSubgroupCiphertextCache();

    PrivateHNSWMetadataV2 metadata_;
    PrivateHNSWConfigV2 config_;

    // Embedding PIR（对称）
    simplepir::EmbeddingPIRParams embPirParams_;
    EmbPIRClient embPirClient_;

    // Neighbor PIR（对称）
    simplepir::NeighborPIRParams nbrPirParams_;
    NbrPIRClient nbrPirClient_;

    // ============================================
    // 预计算池优化
    // ============================================
    std::vector<EmbPIRPrecompute> embPool_;
    size_t embPoolIdx_ = 0;

    std::vector<NbrPIRPrecompute> nbrPool_;
    size_t nbrPoolIdx_ = 0;

    // ============================================
    // 内存池: 预分配的查询向量 (避免重复 malloc)
    // ============================================
    std::vector<std::shared_ptr<EmbMatrix>> embQueryPool_;
    std::vector<std::shared_ptr<NbrMatrix>> nbrQueryPool_;
    size_t embQueryPoolIdx_ = 0;
    size_t nbrQueryPoolIdx_ = 0;

    // 从内存池获取查询向量
    std::shared_ptr<EmbMatrix> getEmbQueryBuffer();
    std::shared_ptr<NbrMatrix> getNbrQueryBuffer();

    // 从池中获取预计算值
    EmbPIRPrecompute getEmbPrecompute();
    NbrPIRPrecompute getNbrPrecompute();

    // 动态池大小估算
    int estimatePoolSize(int ef) const;

    // 历史使用统计
    int lastEmbPoolUsed_ = 0;
    int lastNbrPoolUsed_ = 0;
    double avgEmbPoolUsage_ = 0.0;
    double avgNbrPoolUsage_ = 0.0;
    int queryCount_ = 0;

    // ============================================
    // 预分配缓冲区（避免热路径堆分配）
    // ============================================
    std::vector<int64_t> innerProductsBuf_;

    // ============================================
    // 优化后的缓存结构
    // ============================================
    std::vector<int64_t> distanceCache_;
    std::vector<std::vector<bool>> neighborCache_;
    std::vector<bool> cachedClusters_;

    // 邻居节点ID
    std::unordered_map<int, std::vector<int>> neighborNodeIds_;

    // ============================================
    // 子组级密文缓存（用于选择性解密）
    // ============================================
    struct ClusterCiphertext {
        std::shared_ptr<EmbMatrix> answer;
        std::shared_ptr<EmbMatrix> Hs;
        std::vector<bool> decryptedSubgroups;
        std::vector<int64_t> decryptedDistances;
        bool hasNeighborInfo = false;
        std::vector<std::vector<bool>> neighborClusters;
    };
    std::unordered_map<int, ClusterCiphertext> clusterCiphertextCache_;

    mutable int totalRowsDecrypted_ = 0;
    mutable int totalRowsSkipped_ = 0;

    bool isReady_ = false;
};

/**
 * @brief Utility functions
 */
class PrivateHNSWUtilsV2 {
public:
    static std::shared_ptr<EmbMatrix> generateEmbeddingSharedMatrix(
        const simplepir::EmbeddingPIRParams& params
    );

    static std::shared_ptr<NbrMatrix> generateNeighborSharedMatrix(
        const simplepir::NeighborPIRParams& params
    );
};

} // namespace hnsw

#endif // PRIVATE_HNSW_V2_H

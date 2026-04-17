#pragma once
#include "quantizer.h"
#include "graph_partitioner.h"
#include "subgroup/subgroup_manager.h"
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>
#include <numeric>

namespace hnsw {

enum class DistanceType { L2, InnerProduct };

/**
 * @brief 访问表（用于搜索时标记已访问节点）
 */
class VisitedTable {
public:
    std::vector<uint8_t> visited;
    uint8_t visno;
    int maxSize;
    
    explicit VisitedTable(int size);
    void set(int id);
    bool get(int id) const;
    void reset();
};

/**
 * @brief 量化版 HNSW 索引
 */
class HNSWQuantizedIndex {
public:
    // 基本参数
    int d, M, M0, efConstruction, efSearch, maxLevel;
    std::atomic<int> ntotal;
    float levelMult;
    ScalarQuantizer quantizer;
    DistanceType distType;
    
    // 数据存储
    std::vector<uint8_t> quantizedVectors;
    std::vector<std::vector<std::vector<int32_t>>> neighbors;
    std::vector<int> levels;
    std::atomic<int32_t> entryPoint;
    std::atomic<int> currentMaxLevel;

    // 延迟量化：临时存储原始 float 向量（用于在原始精度下构建图）
    std::vector<float> floatVectors;
    bool useFloatBuild = false;  // 是否使用 float 构建模式
    
    // 随机数生成器
    std::mt19937 rng;
    std::mutex rngMutex;
    
    // 多线程同步
    mutable std::vector<std::mutex> nodeLocks;
    mutable std::mutex globalMutex;
    
    // 聚类相关
    std::vector<int> clusterAssignment;
    int numClusters = 0;
    bool hasClustering = false;

    // ========== 子组相关（新方案）==========
    // 子组划分参数
    int numSubgroupsPerCluster = 0;  // 每个聚类的子组数G
    int maxSubgroupSize = 0;         // 子组最大节点数

    // 节点映射（按聚类+子组顺序重新编号后）
    std::vector<int> nodeToNewId;     // 原始ID -> 新ID
    std::vector<int> newIdToNode;     // 新ID -> 原始ID

    // 边界数组
    std::vector<int> clusterOffset;   // clusterOffset[c] = 聚类c的起始新ID
    std::vector<std::vector<int>> subgroupOffset;  // subgroupOffset[c][g] = 聚类c子组g的起始新ID（相对于聚类）

    // 每个节点的邻居子组信息（直接使用 subgroup_manager.h 中定义的类型，避免类型转换拷贝）
    std::vector<NodeNeighborInfo> nodeNeighborInfo;

    bool hasSubgrouping = false;

    // 直接映射: nodeSubgroup[newId] -> subgroupId (聚类内子组索引)
    // 用于快速查找节点所属子组，避免遍历subgroupOffset
    std::vector<int> nodeSubgroup;

    // 直接映射: nodeLocalIdxInSubgroup[newId] -> 节点在子组内的局部索引
    // BFS分配子组后节点物理位置不变，需要单独记录在子组内的位置
    std::vector<int> nodeLocalIdxInSubgroup;

    // ========== 子组相关方法 ==========
    // 按聚类顺序重新编号节点
    void renumberNodesByCluster();

    // 构建子组划分（满足邻居约束：每个节点的所有邻居都在同一个子组内）
    void buildSubgroups(int targetSubgroupSize = 120);

    // 生成邻居子组信息（用于NeighborPIR）
    void buildNeighborInfo();

    // 序列化子组信息
    void saveSubgroupInfo(std::ofstream& ofs) const;
    void loadSubgroupInfo(std::ifstream& ifs);

    // 获取 SubgroupManager（从现有数据构建）
    SubgroupManager getSubgroupManager() const;
    // Move 版本：零拷贝转移数据到 SubgroupManager（调用后 index 的子组数据失效）
    SubgroupManager moveToSubgroupManager();

    // 节点距离结构
    struct NodeDist {
        int64_t distance;
        int32_t id;
        bool operator<(const NodeDist& o) const { return distance < o.distance; }
        bool operator>(const NodeDist& o) const { return distance > o.distance; }
    };
    
    // 构造函数
    HNSWQuantizedIndex(int dim, int M = 16, int efConstruction = 200,
                       const QuantizerConfig& cfg = QuantizerConfig::uint8Default(),
                       DistanceType distanceType = DistanceType::InnerProduct);
    
    // ========== hnswlib 加速构建 ==========
#ifdef USE_HNSWLIB
    // 使用 hnswlib 构建图 + 量化（一步完成，无需 finalizeQuantization）
    void buildWithHnswlib(const float* data, int n, int numThreads = 0);
    void buildWithHnswlibMove(std::vector<float>&& data, int n, int numThreads = 0);
#endif

    // ========== 延迟量化模式：在原始 float 精度下构建图 ==========
    // 使用 float 向量构建图（不立即量化）
    void addFloatParallel(const float* data, int n, int numThreads = 0);
    void addFloatParallelMove(std::vector<float>&& data, int n, int numThreads = 0);
    // 在构建完成后进行量化（将 floatVectors 量化为 quantizedVectors）
    void finalizeQuantization();
    // Float 距离计算（用于构建阶段）
    float computeDistanceFloat(int32_t id, const float* query) const;

    // ========== 搜索 ==========
    std::vector<std::pair<int64_t, int32_t>> search(const float* query, int k, int ef = -1) const;
    std::vector<std::pair<int64_t, int32_t>> searchQuantized(const uint8_t* query, int k, int ef = -1) const;
    
    // ========== 距离计算 ==========
    int64_t computeDistance(int32_t id, const uint8_t* query) const;
    
    // ========== 聚类 ==========
    void buildClustering(int nClusters = -1, int numThreads = -1);
    void buildGraphPartitionClustering(int nClusters = -1, double imbalance = 0.05);
    bool hasClusteringData() const { return hasClustering; }
    int getNumClusters() const { return numClusters; }
    int getClusterAssignment(int nodeId) const;

    // 聚类分析
    struct ClusteringStats {
        int numClusters;
        int maxClusterSize;
        int minClusterSize;
        double avgClusterSize;
        double imbalance;
        double avgNeighborClusters;
        double edgeCutRatio;
    };
    ClusteringStats analyzeCurrentClustering() const;
    
    // ========== 序列化 ==========
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    int randomLevel();

    // 搜索辅助函数（量化版本）
    int32_t greedySearch(const uint8_t* query, int32_t entry, int level) const;
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> searchLayer(const uint8_t* query, int32_t entry, int ef, int level, VisitedTable& vt) const;

    // ========== Float 版本的辅助函数（用于延迟量化模式）==========
    // Float 距离节点结构
    struct NodeDistFloat {
        float distance;
        int32_t id;
        bool operator<(const NodeDistFloat& o) const { return distance < o.distance; }
        bool operator>(const NodeDistFloat& o) const { return distance > o.distance; }
    };

    // Float 搜索辅助函数
    int32_t greedySearchFloat(const float* query, int32_t entry, int level) const;
    std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>>
        searchLayerFloat(const float* query, int32_t entry, int ef, int level, VisitedTable& vt) const;

    // Float 图构建辅助函数
    std::vector<int32_t> selectNeighborsFloat(const float* query,
        std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>>& candidates, int maxM) const;
    void addConnectionFloatThreadSafe(int32_t from, int32_t to, int level);
    void addOneFloatParallel(int32_t id, int level, int totalNodes);

    // 聚类辅助函数已迁移到 clustering/clustering.h

#ifdef USE_HNSWLIB
    // hnswlib 图提取
    void extractGraphFromHnswlib(void* hnswPtr, int n);
#endif
};

} // namespace hnsw

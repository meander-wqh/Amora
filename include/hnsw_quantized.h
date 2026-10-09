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

class HNSWQuantizedIndex {
public:
    int d, M, M0, efConstruction, efSearch, maxLevel;
    std::atomic<int> ntotal;
    float levelMult;
    ScalarQuantizer quantizer;
    DistanceType distType;
    
    std::vector<uint8_t> quantizedVectors;
    std::vector<std::vector<std::vector<int32_t>>> neighbors;
    std::vector<int> levels;
    std::atomic<int32_t> entryPoint;
    std::atomic<int> currentMaxLevel;

    std::vector<float> floatVectors;
    bool useFloatBuild = false;
    
    std::mt19937 rng;
    std::mutex rngMutex;
    
    mutable std::vector<std::mutex> nodeLocks;
    mutable std::mutex globalMutex;
    
    std::vector<int> clusterAssignment;
    int numClusters = 0;
    bool hasClustering = false;

    int numSubgroupsPerCluster = 0;
    int maxSubgroupSize = 0;

    std::vector<int> nodeToNewId;
    std::vector<int> newIdToNode;

    std::vector<int> clusterOffset;
    std::vector<std::vector<int>> subgroupOffset;

    std::vector<NodeNeighborInfo> nodeNeighborInfo;

    bool hasSubgrouping = false;

    std::vector<int> nodeSubgroup;

    std::vector<int> nodeLocalIdxInSubgroup;

    void renumberNodesByCluster();

    void buildSubgroups(int targetSubgroupSize = 120);

    void buildNeighborInfo();

    void saveSubgroupInfo(std::ofstream& ofs) const;
    void loadSubgroupInfo(std::ifstream& ifs);

    SubgroupManager getSubgroupManager() const;
    SubgroupManager moveToSubgroupManager();

    struct NodeDist {
        int64_t distance;
        int32_t id;
        bool operator<(const NodeDist& o) const { return distance < o.distance; }
        bool operator>(const NodeDist& o) const { return distance > o.distance; }
    };
    
    HNSWQuantizedIndex(int dim, int M = 16, int efConstruction = 200,
                       const QuantizerConfig& cfg = QuantizerConfig::uint8Default(),
                       DistanceType distanceType = DistanceType::InnerProduct);
    
#ifdef USE_HNSWLIB
    void buildWithHnswlib(const float* data, int n, int numThreads = 0);
    void buildWithHnswlibMove(std::vector<float>&& data, int n, int numThreads = 0);
#endif

    void addFloatParallel(const float* data, int n, int numThreads = 0);
    void addFloatParallelMove(std::vector<float>&& data, int n, int numThreads = 0);
    void finalizeQuantization();
    float computeDistanceFloat(int32_t id, const float* query) const;

    std::vector<std::pair<int64_t, int32_t>> search(const float* query, int k, int ef = -1) const;
    std::vector<std::pair<int64_t, int32_t>> searchQuantized(const uint8_t* query, int k, int ef = -1) const;
    
    int64_t computeDistance(int32_t id, const uint8_t* query) const;
    
    void buildClustering(int nClusters = -1, int numThreads = -1);
    void buildGraphPartitionClustering(int nClusters = -1, double imbalance = 0.05);
    bool hasClusteringData() const { return hasClustering; }
    int getNumClusters() const { return numClusters; }
    int getClusterAssignment(int nodeId) const;

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
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    int randomLevel();

    int32_t greedySearch(const uint8_t* query, int32_t entry, int level) const;
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> searchLayer(const uint8_t* query, int32_t entry, int ef, int level, VisitedTable& vt) const;

    struct NodeDistFloat {
        float distance;
        int32_t id;
        bool operator<(const NodeDistFloat& o) const { return distance < o.distance; }
        bool operator>(const NodeDistFloat& o) const { return distance > o.distance; }
    };

    int32_t greedySearchFloat(const float* query, int32_t entry, int level) const;
    std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>>
        searchLayerFloat(const float* query, int32_t entry, int ef, int level, VisitedTable& vt) const;

    std::vector<int32_t> selectNeighborsFloat(const float* query,
        std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>>& candidates, int maxM) const;
    void addConnectionFloatThreadSafe(int32_t from, int32_t to, int level);
    void addOneFloatParallel(int32_t id, int level, int totalNodes);

#ifdef USE_HNSWLIB
    void extractGraphFromHnswlib(void* hnswPtr, int n);
#endif
};

}

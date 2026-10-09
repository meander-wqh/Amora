#pragma once

#include <vector>
#include <memory>
#include <string>

namespace hnsw {

class HNSWQuantizedIndex;

struct ClusteringResult {
    std::vector<int> assignments;
    std::vector<std::vector<int>> clusters;
    int numClusters = 0;

    int minClusterSize = 0;
    int maxClusterSize = 0;
    double avgClusterSize = 0.0;
    double imbalance = 0.0;
    double edgeCutRatio = 0.0;
    double avgNeighborClusters = 0.0;

    void buildClustersFromAssignments();

    void print() const;
};

class ClusteringStrategy {
public:
    virtual ~ClusteringStrategy() = default;

    virtual ClusteringResult cluster(const HNSWQuantizedIndex& index,
                                     int numClusters) = 0;

    virtual std::string name() const = 0;
};

class KMeansClustering : public ClusteringStrategy {
public:
    explicit KMeansClustering(int maxIter = 20, int numThreads = 0);

    ClusteringResult cluster(const HNSWQuantizedIndex& index,
                            int numClusters) override;

    std::string name() const override { return "KMeans"; }

private:
    int maxIter_;
    int numThreads_;

    void kMeansPlusPlusInit(const HNSWQuantizedIndex& index,
                            std::vector<std::vector<float>>& centroids,
                            int k);

    float computeL2DistToCentroid(const uint8_t* vec, int dim,
                                  const std::vector<float>& centroid) const;

    int findNearestCentroid(const uint8_t* vec, int dim,
                           const std::vector<std::vector<float>>& centroids) const;
};

class GraphPartitionClustering : public ClusteringStrategy {
public:
    explicit GraphPartitionClustering(double imbalance = 0.05);

    ClusteringResult cluster(const HNSWQuantizedIndex& index,
                            int numClusters) override;

    std::string name() const override { return "GraphPartition"; }

    static int computeOptimalClusters(int n, int dim,
                                      int alpha = 5, double beta = 1.5,
                                      double gamma = 1.5);

private:
    double imbalance_;
};

class ClusteringFactory {
public:
    enum class Method {
        KMeans,
        GraphPartition
    };

    static std::unique_ptr<ClusteringStrategy> create(Method method);

    static std::unique_ptr<ClusteringStrategy> create(const std::string& methodName);
};

class ClusteringAnalyzer {
public:
    static ClusteringResult analyze(const HNSWQuantizedIndex& index,
                                   const std::vector<int>& assignments,
                                   int numClusters);
};

}

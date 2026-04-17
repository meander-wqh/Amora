#pragma once

#include <vector>
#include <memory>
#include <string>

namespace hnsw {

// 前向声明
class HNSWQuantizedIndex;

/**
 * @brief 聚类结果
 */
struct ClusteringResult {
    std::vector<int> assignments;           // 每个节点的聚类ID (大小 = N)
    std::vector<std::vector<int>> clusters; // 每个聚类包含的节点列表
    int numClusters = 0;

    // 统计信息
    int minClusterSize = 0;
    int maxClusterSize = 0;
    double avgClusterSize = 0.0;
    double imbalance = 0.0;         // (max - avg) / avg
    double edgeCutRatio = 0.0;      // 跨聚类边的比例
    double avgNeighborClusters = 0.0;

    // 从分配向量构建聚类列表
    void buildClustersFromAssignments();

    // 打印统计信息
    void print() const;
};

/**
 * @brief 聚类策略接口
 *
 * 使用策略模式，支持多种聚类算法
 */
class ClusteringStrategy {
public:
    virtual ~ClusteringStrategy() = default;

    /**
     * @brief 执行聚类
     *
     * @param index HNSW 索引（包含向量和图结构）
     * @param numClusters 目标聚类数量
     * @return 聚类结果
     */
    virtual ClusteringResult cluster(const HNSWQuantizedIndex& index,
                                     int numClusters) = 0;

    /**
     * @brief 获取策略名称
     */
    virtual std::string name() const = 0;
};

/**
 * @brief K-Means 聚类策略
 *
 * 使用 K-Means++ 初始化，支持 SIMD 加速
 */
class KMeansClustering : public ClusteringStrategy {
public:
    /**
     * @param maxIter 最大迭代次数
     * @param numThreads 线程数（0 = 自动）
     */
    explicit KMeansClustering(int maxIter = 20, int numThreads = 0);

    ClusteringResult cluster(const HNSWQuantizedIndex& index,
                            int numClusters) override;

    std::string name() const override { return "KMeans"; }

private:
    int maxIter_;
    int numThreads_;

    // K-Means++ 初始化
    void kMeansPlusPlusInit(const HNSWQuantizedIndex& index,
                            std::vector<std::vector<float>>& centroids,
                            int k);

    // 计算向量到质心的 L2 距离（支持 SIMD）
    float computeL2DistToCentroid(const uint8_t* vec, int dim,
                                  const std::vector<float>& centroid) const;

    // 找到最近的质心
    int findNearestCentroid(const uint8_t* vec, int dim,
                           const std::vector<std::vector<float>>& centroids) const;
};

/**
 * @brief 图分区聚类策略
 *
 * 使用 METIS 风格的多级图分区算法
 */
class GraphPartitionClustering : public ClusteringStrategy {
public:
    /**
     * @param imbalance 允许的不平衡度（如 0.05 表示 5%）
     */
    explicit GraphPartitionClustering(double imbalance = 0.05);

    ClusteringResult cluster(const HNSWQuantizedIndex& index,
                            int numClusters) override;

    std::string name() const override { return "GraphPartition"; }

    /**
     * @brief 计算最优聚类数量
     *
     * 公式: C* = sqrt(N * gamma * (1 + (alpha + 1) * beta) / (d + 1))
     */
    static int computeOptimalClusters(int n, int dim,
                                      int alpha = 5, double beta = 1.5,
                                      double gamma = 1.5);

private:
    double imbalance_;
};

/**
 * @brief 聚类策略工厂
 *
 * 根据方法名称创建相应的策略对象
 */
class ClusteringFactory {
public:
    enum class Method {
        KMeans,
        GraphPartition
    };

    /**
     * @brief 创建聚类策略
     *
     * @param method 聚类方法
     * @return 策略对象
     */
    static std::unique_ptr<ClusteringStrategy> create(Method method);

    /**
     * @brief 从字符串创建聚类策略
     *
     * @param methodName 方法名称 ("kmeans" 或 "graphpartition")
     * @return 策略对象
     */
    static std::unique_ptr<ClusteringStrategy> create(const std::string& methodName);
};

/**
 * @brief 聚类分析工具
 *
 * 分析聚类质量和统计信息
 */
class ClusteringAnalyzer {
public:
    /**
     * @brief 分析现有聚类
     *
     * @param index HNSW 索引
     * @param assignments 聚类分配
     * @param numClusters 聚类数量
     * @return 统计信息
     */
    static ClusteringResult analyze(const HNSWQuantizedIndex& index,
                                   const std::vector<int>& assignments,
                                   int numClusters);
};

} // namespace hnsw

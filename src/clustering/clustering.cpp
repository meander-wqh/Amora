#include "clustering.h"
#include "hnsw_quantized.h"
#include "graph_partitioner.h"

#ifdef USE_MTMETIS
extern "C" {
#include "mtmetis.h"
}
#endif

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <mutex>
#include <set>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#include <pmmintrin.h>
#endif

namespace hnsw {

// ============================================================================
// ClusteringResult
// ============================================================================

void ClusteringResult::buildClustersFromAssignments() {
    if (assignments.empty() || numClusters <= 0) return;

    clusters.clear();
    clusters.resize(numClusters);

    for (size_t i = 0; i < assignments.size(); ++i) {
        int c = assignments[i];
        if (c >= 0 && c < numClusters) {
            clusters[c].push_back(static_cast<int>(i));
        }
    }

    // 计算统计信息
    if (numClusters > 0) {
        std::vector<int> sizes(numClusters);
        for (int c = 0; c < numClusters; ++c) {
            sizes[c] = static_cast<int>(clusters[c].size());
        }
        minClusterSize = *std::min_element(sizes.begin(), sizes.end());
        maxClusterSize = *std::max_element(sizes.begin(), sizes.end());
        avgClusterSize = static_cast<double>(assignments.size()) / numClusters;
        imbalance = (avgClusterSize > 0) ?
            (maxClusterSize - avgClusterSize) / avgClusterSize : 0.0;
    }
}

void ClusteringResult::print() const {
    std::cout << "=== Clustering Result ===" << std::endl;
    std::cout << "  Num clusters: " << numClusters << std::endl;
    std::cout << "  Cluster sizes: min=" << minClusterSize
              << ", avg=" << std::fixed << std::setprecision(1) << avgClusterSize
              << ", max=" << maxClusterSize << std::endl;
    std::cout << "  Imbalance: " << std::fixed << std::setprecision(2)
              << (imbalance * 100) << "%" << std::endl;
    if (edgeCutRatio > 0 || avgNeighborClusters > 0) {
        std::cout << "  Edge cut ratio: " << std::fixed << std::setprecision(2)
                  << (edgeCutRatio * 100) << "%" << std::endl;
        std::cout << "  Avg neighbor clusters: " << std::fixed << std::setprecision(2)
                  << avgNeighborClusters << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

// ============================================================================
// KMeansClustering
// ============================================================================

KMeansClustering::KMeansClustering(int maxIter, int numThreads)
    : maxIter_(maxIter), numThreads_(numThreads) {}

float KMeansClustering::computeL2DistToCentroid(
    const uint8_t* vec, int dim, const std::vector<float>& centroid) const {

    float dist = 0.0f;

#if defined(__AVX2__)
    // AVX2 优化版本
    int j = 0;
    __m256 sum = _mm256_setzero_ps();

    for (; j + 7 < dim; j += 8) {
        __m256 v = _mm256_set_ps(
            (float)vec[j+7], (float)vec[j+6], (float)vec[j+5], (float)vec[j+4],
            (float)vec[j+3], (float)vec[j+2], (float)vec[j+1], (float)vec[j]
        );
        __m256 c = _mm256_loadu_ps(&centroid[j]);
        __m256 diff = _mm256_sub_ps(v, c);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // 水平求和
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    dist = _mm_cvtss_f32(sum128);

    // 处理剩余元素
    for (; j < dim; j++) {
        float diff = (float)vec[j] - centroid[j];
        dist += diff * diff;
    }
#elif defined(__SSE2__)
    // SSE2 优化版本
    int j = 0;
    __m128 sum = _mm_setzero_ps();

    for (; j + 3 < dim; j += 4) {
        __m128 v = _mm_set_ps(
            (float)vec[j+3], (float)vec[j+2], (float)vec[j+1], (float)vec[j]
        );
        __m128 c = _mm_loadu_ps(&centroid[j]);
        __m128 diff = _mm_sub_ps(v, c);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // 水平求和
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    dist = _mm_cvtss_f32(sum);

    // 处理剩余元素
    for (; j < dim; j++) {
        float diff = (float)vec[j] - centroid[j];
        dist += diff * diff;
    }
#else
    // 标量版本
    for (int j = 0; j < dim; j++) {
        float diff = (float)vec[j] - centroid[j];
        dist += diff * diff;
    }
#endif

    return dist;
}

int KMeansClustering::findNearestCentroid(
    const uint8_t* vec, int dim,
    const std::vector<std::vector<float>>& centroids) const {

    int k = static_cast<int>(centroids.size());
    int bestCluster = 0;
    float bestDist = std::numeric_limits<float>::max();

    for (int c = 0; c < k; c++) {
        float dist = computeL2DistToCentroid(vec, dim, centroids[c]);
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
        }
    }
    return bestCluster;
}

void KMeansClustering::kMeansPlusPlusInit(
    const HNSWQuantizedIndex& index,
    std::vector<std::vector<float>>& centroids,
    int k) {

    int n = index.ntotal.load();
    int d = index.d;
    if (n == 0 || k <= 0) return;

    std::vector<float> minDistSq(n, std::numeric_limits<float>::max());

    // 随机数生成器
    std::mt19937 rng(42);

    // 随机选择第一个中心
    std::uniform_int_distribution<int> dist(0, n - 1);
    int firstIdx = dist(rng);

    const uint8_t* firstVec = index.quantizedVectors.data() + (size_t)firstIdx * d;
    for (int j = 0; j < d; j++) {
        centroids[0][j] = (float)firstVec[j];
    }

    std::cout << "K-means++ initializing " << k << " centroids..." << std::endl;

    for (int c = 1; c < k; c++) {
        // 并行计算每个点到最近已选中心的距离
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < n; i++) {
            const uint8_t* vec = index.quantizedVectors.data() + (size_t)i * d;
            float distVal = computeL2DistToCentroid(vec, d, centroids[c - 1]);
            if (distVal < minDistSq[i]) {
                minDistSq[i] = distVal;
            }
        }

        // 计算累积概率分布
        double totalDistSq = 0.0;
        for (int i = 0; i < n; i++) {
            totalDistSq += minDistSq[i];
        }

        // 按概率选择下一个中心
        std::uniform_real_distribution<double> realDist(0.0, totalDistSq);
        double r = realDist(rng);

        double cumSum = 0.0;
        int nextIdx = n - 1;
        for (int i = 0; i < n; i++) {
            cumSum += minDistSq[i];
            if (cumSum >= r) {
                nextIdx = i;
                break;
            }
        }

        const uint8_t* nextVec = index.quantizedVectors.data() + (size_t)nextIdx * d;
        for (int j = 0; j < d; j++) {
            centroids[c][j] = (float)nextVec[j];
        }

        if ((c + 1) % 100 == 0 || c + 1 == k) {
            std::cout << "  Initialized " << (c + 1) << "/" << k << " centroids" << std::endl;
        }
    }
}

ClusteringResult KMeansClustering::cluster(
    const HNSWQuantizedIndex& index, int numClusters) {

    ClusteringResult result;
    int n = index.ntotal.load();
    int d = index.d;

    if (n == 0 || numClusters <= 0) return result;

    result.numClusters = numClusters;
    result.assignments.resize(n, 0);

#ifdef _OPENMP
    if (numThreads_ > 0) {
        omp_set_num_threads(numThreads_);
        std::cout << "Using " << numThreads_ << " threads for K-Means" << std::endl;
    } else {
        std::cout << "Using " << omp_get_max_threads() << " threads for K-Means" << std::endl;
    }
#endif

    std::vector<std::vector<float>> centroids(numClusters, std::vector<float>(d, 0.0f));

    // K-means++ 初始化
    kMeansPlusPlusInit(index, centroids, numClusters);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIter_; iter++) {
        // E步: 并行分配每个点到最近的聚类中心
        std::atomic<int> changed(0);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < n; i++) {
            const uint8_t* vec = index.quantizedVectors.data() + (size_t)i * d;
            int bestCluster = findNearestCentroid(vec, d, centroids);

            if (result.assignments[i] != bestCluster) {
                result.assignments[i] = bestCluster;
                changed.fetch_add(1, std::memory_order_relaxed);
            }
        }

        auto iterTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(iterTime - startTime).count();
        std::cout << "K-Means iteration " << iter << ": " << changed.load() << " changed, "
                  << std::fixed << std::setprecision(1) << elapsed << "s" << std::endl;

        if (changed.load() == 0) {
            std::cout << "K-Means converged at iteration " << iter << std::endl;
            break;
        }

        // M步: 更新聚类中心
        #ifdef _OPENMP
            int actualThreads = (numThreads_ > 0) ? numThreads_ : omp_get_max_threads();
        #else
            int actualThreads = 1;
        #endif

        std::vector<std::vector<std::vector<double>>> localSums(actualThreads,
            std::vector<std::vector<double>>(numClusters, std::vector<double>(d, 0.0)));
        std::vector<std::vector<int>> localCounts(actualThreads, std::vector<int>(numClusters, 0));

        #ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                int c = result.assignments[i];
                localCounts[tid][c]++;
                const uint8_t* vec = index.quantizedVectors.data() + (size_t)i * d;
                for (int j = 0; j < d; j++) {
                    localSums[tid][c][j] += vec[j];
                }
            }
        }
        #else
        for (int i = 0; i < n; i++) {
            int c = result.assignments[i];
            localCounts[0][c]++;
            const uint8_t* vec = index.quantizedVectors.data() + (size_t)i * d;
            for (int j = 0; j < d; j++) {
                localSums[0][c][j] += vec[j];
            }
        }
        #endif

        // 合并结果
        std::vector<int> counts(numClusters, 0);
        for (int c = 0; c < numClusters; c++) {
            std::fill(centroids[c].begin(), centroids[c].end(), 0.0f);
            for (int t = 0; t < actualThreads; t++) {
                counts[c] += localCounts[t][c];
                for (int j = 0; j < d; j++) {
                    centroids[c][j] += (float)localSums[t][c][j];
                }
            }
            if (counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    centroids[c][j] /= counts[c];
                }
            }
        }
    }

    // 构建聚类列表
    result.buildClustersFromAssignments();

    return result;
}

// ============================================================================
// GraphPartitionClustering
// ============================================================================

GraphPartitionClustering::GraphPartitionClustering(double imbalance)
    : imbalance_(imbalance) {}

int GraphPartitionClustering::computeOptimalClusters(
    int n, int dim, int alpha, double beta, double gamma) {
    // C* = sqrt(N * gamma * (1 + (alpha + 1) * beta) / (d + 1))
    double factor = gamma * (1.0 + (alpha + 1) * beta) / (dim + 1);
    return static_cast<int>(std::sqrt(n * factor));
}

ClusteringResult GraphPartitionClustering::cluster(
    const HNSWQuantizedIndex& index, int numClusters) {

    ClusteringResult result;
    int n = index.ntotal.load();

    if (n == 0) return result;

    if (numClusters <= 0) {
        // 默认: sqrt(N/d) 个聚类（基于EmbeddingPIR维度优化）
        numClusters = std::max(1, (int)std::sqrt((double)n / index.d));
    }
    result.numClusters = numClusters;

    std::cout << "Building graph partition clustering with " << numClusters
              << " clusters for " << n << " vectors..." << std::endl;

#ifdef USE_MTMETIS
    // ===================== MT-METIS 路径 =====================
    std::cout << "Using MT-METIS for multi-threaded graph partitioning..." << std::endl;

    // HNSW 图可能不完全对称（pruning 导致），MT-METIS 要求对称 CSR
    // 先构建对称邻接表，去除自环
    std::vector<std::vector<mtmetis_vtx_type>> symAdj(n);
    for (int i = 0; i < n; ++i) {
        if (!index.neighbors[i].empty()) {
            for (int32_t nbr : index.neighbors[i][0]) {
                if (nbr != i && nbr >= 0 && nbr < n) {
                    symAdj[i].push_back(static_cast<mtmetis_vtx_type>(nbr));
                    symAdj[nbr].push_back(static_cast<mtmetis_vtx_type>(i));
                }
            }
        }
    }
    // 去重排序
    for (int i = 0; i < n; ++i) {
        auto& adj = symAdj[i];
        std::sort(adj.begin(), adj.end());
        adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
    }

    // 转为 CSR
    std::vector<mtmetis_adj_type> xadj(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        xadj[i + 1] = xadj[i] + symAdj[i].size();
    }
    std::vector<mtmetis_vtx_type> adjncy(xadj[n]);
    for (int i = 0; i < n; ++i) {
        std::copy(symAdj[i].begin(), symAdj[i].end(), adjncy.begin() + xadj[i]);
    }
    // 释放临时邻接表
    { std::vector<std::vector<mtmetis_vtx_type>>().swap(symAdj); }

    std::cout << "  CSR graph: " << n << " vertices, " << xadj[n] << " directed edges" << std::endl;

    // 配置 MT-METIS 选项
    double* options = mtmetis_init_options();
    int numThreads = 0;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#else
    numThreads = 1;
#endif
    options[MTMETIS_OPTION_NTHREADS] = numThreads;
    options[MTMETIS_OPTION_SEED] = 42;
    options[MTMETIS_OPTION_NCUTS] = 1;
    options[MTMETIS_OPTION_NITER] = 10;

    // 不平衡因子
    mtmetis_vtx_type nvtxs = n;
    mtmetis_vtx_type ncon = 1;
    mtmetis_pid_type nparts = numClusters;
    mtmetis_real_type ubvec = 1.0f + static_cast<float>(imbalance_);
    mtmetis_wgt_type edgecut = 0;
    std::vector<mtmetis_pid_type> where(n);

    auto startTime = std::chrono::high_resolution_clock::now();

    int ret = MTMETIS_PartGraphKway(
        &nvtxs, &ncon,
        xadj.data(), adjncy.data(),
        NULL, NULL, NULL,   // vwgt, vsize, adjwgt (全部无权重)
        &nparts,
        NULL,               // tpwgts (等分)
        &ubvec,
        options,
        &edgecut,
        where.data()
    );

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    free(options);

    if (ret != MTMETIS_SUCCESS) {
        std::cerr << "MT-METIS partitioning failed with error code " << ret << std::endl;
        throw std::runtime_error("MT-METIS partitioning failed");
    }

    std::cout << "MT-METIS partitioning completed in " << std::fixed << std::setprecision(2)
              << elapsed << "s, edge cut: " << edgecut
              << ", threads: " << numThreads << std::endl;

    // 转换结果
    result.assignments.resize(n);
    for (int i = 0; i < n; ++i) {
        result.assignments[i] = static_cast<int>(where[i]);
    }

#else
    // ===================== 自研分区器路径 =====================
    // 提取 level 0 邻居
    std::vector<std::vector<int>> level0Neighbors(n);
    for (int i = 0; i < n; i++) {
        if (!index.neighbors[i].empty()) {
            for (int32_t nbr : index.neighbors[i][0]) {
                level0Neighbors[i].push_back(nbr);
            }
        }
    }

    // 配置分区器
    PartitionConfig config;
    config.numParts = numClusters;
    config.imbalance = imbalance_;
    config.verbose = true;

    // 执行图分区
    auto startTime = std::chrono::high_resolution_clock::now();
    HNSWGraphPartitioner partitioner(config);
    auto partResult = partitioner.partition(n, level0Neighbors);
    auto endTime = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "Graph partitioning time: " << std::fixed << std::setprecision(2)
              << elapsed << " s" << std::endl;

    // 存储结果
    result.assignments = partResult.partition;
#endif

    // 构建聚类列表
    result.buildClustersFromAssignments();

    // 分析图聚类质量
    result = ClusteringAnalyzer::analyze(index, result.assignments, numClusters);

    return result;
}

// ============================================================================
// ClusteringFactory
// ============================================================================

std::unique_ptr<ClusteringStrategy> ClusteringFactory::create(Method method) {
    switch (method) {
        case Method::KMeans:
            return std::make_unique<KMeansClustering>();
        case Method::GraphPartition:
            return std::make_unique<GraphPartitionClustering>();
        default:
            return std::make_unique<GraphPartitionClustering>();
    }
}

std::unique_ptr<ClusteringStrategy> ClusteringFactory::create(const std::string& methodName) {
    std::string lower = methodName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "kmeans" || lower == "k-means") {
        return create(Method::KMeans);
    } else if (lower == "graphpartition" || lower == "graph_partition" || lower == "metis") {
        return create(Method::GraphPartition);
    }

    // 默认使用图分区
    return create(Method::GraphPartition);
}

// ============================================================================
// ClusteringAnalyzer
// ============================================================================

ClusteringResult ClusteringAnalyzer::analyze(
    const HNSWQuantizedIndex& index,
    const std::vector<int>& assignments,
    int numClusters) {

    ClusteringResult result;
    result.assignments = assignments;
    result.numClusters = numClusters;

    int n = index.ntotal.load();
    if (n == 0 || numClusters <= 0) return result;

    // 构建聚类列表
    result.buildClustersFromAssignments();

    // 分析邻居聚类和边切割
    double totalNeighborClusters = 0;
    int totalEdges = 0;
    int edgeCut = 0;

    for (int u = 0; u < n; u++) {
        std::set<int> neighborClusters;
        if (!index.neighbors[u].empty()) {
            for (int32_t v : index.neighbors[u][0]) {
                if (v >= 0 && v < n) {
                    neighborClusters.insert(assignments[v]);
                    totalEdges++;
                    if (assignments[u] != assignments[v]) {
                        edgeCut++;
                    }
                }
            }
        }
        totalNeighborClusters += neighborClusters.size();
    }

    result.avgNeighborClusters = totalNeighborClusters / n;
    result.edgeCutRatio = (totalEdges > 0) ?
        static_cast<double>(edgeCut) / totalEdges : 0;

    return result;
}

} // namespace hnsw

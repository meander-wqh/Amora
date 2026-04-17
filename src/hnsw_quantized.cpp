#include "hnsw_quantized.h"
#include "clustering/clustering.h"
#include <cassert>
#include <limits>
#include <set>
#include <queue>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_HNSWLIB
#include "hnswlib/hnswlib.h"
#endif

// SIMD headers
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <pmmintrin.h>  // for _mm_hadd_ps
#endif

namespace hnsw {

// ========== VisitedTable 实现 ==========

VisitedTable::VisitedTable(int size) : visited(size, 0), visno(1), maxSize(size) {}

void VisitedTable::set(int id) {
    if (id >= 0 && id < maxSize) visited[id] = visno;
}

bool VisitedTable::get(int id) const {
    return id >= 0 && id < maxSize && visited[id] == visno;
}

void VisitedTable::reset() {
    if (++visno == 0) {
        std::fill(visited.begin(), visited.end(), 0);
        visno = 1;
    }
}

// ========== HNSWQuantizedIndex 构造函数 ==========

HNSWQuantizedIndex::HNSWQuantizedIndex(int dim, int M, int efConstruction,
                                       const QuantizerConfig& cfg, DistanceType distanceType)
    : d(dim), M(M), M0(M * 2), efConstruction(efConstruction), efSearch(16),
      maxLevel(16), ntotal(0), levelMult(1.0f / std::log(M)),
      quantizer(dim, cfg), distType(distanceType),
      entryPoint(-1), currentMaxLevel(-1), rng(std::random_device{}()) {}

// ========== 距离计算 ==========

int64_t HNSWQuantizedIndex::computeDistance(int32_t id, const uint8_t* query) const {
    const uint8_t* vec = quantizedVectors.data() + (size_t)id * d;
    int64_t dist = 0;

    if (quantizer.config.isUnsigned) {
        if (distType == DistanceType::L2) {
            for (int i = 0; i < d; ++i) {
                int diff = (int)vec[i] - (int)query[i];
                dist += diff * diff;
            }
        } else {
            for (int i = 0; i < d; ++i) {
                dist -= (int64_t)vec[i] * (int64_t)query[i];
            }
        }
    } else {
        // signed量化：将uint8_t重新解释为int8_t
        const int8_t* svec = reinterpret_cast<const int8_t*>(vec);
        const int8_t* squery = reinterpret_cast<const int8_t*>(query);
        if (distType == DistanceType::L2) {
            for (int i = 0; i < d; ++i) {
                int diff = (int)svec[i] - (int)squery[i];
                dist += diff * diff;
            }
        } else {
            for (int i = 0; i < d; ++i) {
                dist -= (int64_t)svec[i] * (int64_t)squery[i];
            }
        }
    }
    return dist;
}

// ========== 随机层数 ==========

int HNSWQuantizedIndex::randomLevel() {
    std::lock_guard<std::mutex> lock(rngMutex);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    int level = (int)(-std::log(r) * levelMult);
    return std::min(level, maxLevel - 1);
}

// ========== 贪婪搜索 ==========

int32_t HNSWQuantizedIndex::greedySearch(const uint8_t* query, int32_t entry, int level) const {
    int32_t curr = entry;
    int64_t currDist = computeDistance(curr, query);
    
    bool changed = true;
    while (changed) {
        changed = false;
        // 边界检查
        if (curr < 0 || curr >= (int)neighbors.size()) break;
        if (level < 0 || level >= (int)neighbors[curr].size()) break;
        
        const auto& nbrs = neighbors[curr][level];
        for (int32_t nbr : nbrs) {
            if (nbr < 0 || nbr >= (int)neighbors.size()) continue;
            int64_t d = computeDistance(nbr, query);
            if (d < currDist) {
                currDist = d;
                curr = nbr;
                changed = true;
            }
        }
    }
    return curr;
}


// ========== 层内搜索 ==========

std::priority_queue<HNSWQuantizedIndex::NodeDist, std::vector<HNSWQuantizedIndex::NodeDist>, std::greater<HNSWQuantizedIndex::NodeDist>>
HNSWQuantizedIndex::searchLayer(const uint8_t* query, int32_t entry, int ef, int level, VisitedTable& vt) const {
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> candidates;
    std::priority_queue<NodeDist> results;
    
    int64_t d = computeDistance(entry, query);
    candidates.push({d, entry});
    results.push({d, entry});
    vt.set(entry);
    
    while (!candidates.empty()) {
        NodeDist curr = candidates.top();
        candidates.pop();
        
        if (curr.distance > results.top().distance) break;
        
        // 边界检查
        if (curr.id < 0 || curr.id >= (int)neighbors.size()) continue;
        if (level < 0 || level >= (int)neighbors[curr.id].size()) continue;
        
        const auto& nbrs = neighbors[curr.id][level];
        for (int32_t nbr : nbrs) {
            if (nbr < 0 || nbr >= (int)neighbors.size() || vt.get(nbr)) continue;
            vt.set(nbr);
            
            int64_t nd = computeDistance(nbr, query);
            if (nd < results.top().distance || (int)results.size() < ef) {
                candidates.push({nd, nbr});
                results.push({nd, nbr});
                if ((int)results.size() > ef) results.pop();
            }
        }
    }
    
    // 转换为min heap返回
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> ret;
    while (!results.empty()) {
        ret.push(results.top());
        results.pop();
    }
    return ret;
}









// ========== 搜索 ==========

std::vector<std::pair<int64_t, int32_t>> HNSWQuantizedIndex::search(const float* query, int k, int ef) const {
    if (ef < 0) ef = std::max(efSearch, k);
    
    std::vector<uint8_t> qVec(d);
    quantizer.quantize(query, qVec.data(), 1);
    
    return searchQuantized(qVec.data(), k, ef);
}

std::vector<std::pair<int64_t, int32_t>> HNSWQuantizedIndex::searchQuantized(const uint8_t* query, int k, int ef) const {
    if (ntotal.load() == 0) return {};
    if (ef < 0) ef = std::max(efSearch, k);
    
    int32_t curr = entryPoint.load();
    int curMaxLevel = currentMaxLevel.load();
    
    // 上层贪婪搜索
    for (int lev = curMaxLevel; lev > 0; --lev) {
        curr = greedySearch(query, curr, lev);
    }
    
    // 底层beam search
    VisitedTable vt(ntotal.load());
    auto candidates = searchLayer(query, curr, ef, 0, vt);
    
    // 取top-k
    std::vector<std::pair<int64_t, int32_t>> results;
    while (!candidates.empty() && (int)results.size() < k) {
        results.push_back({candidates.top().distance, candidates.top().id});
        candidates.pop();
    }
    
    return results;
}


// ========== 聚类 ==========

void HNSWQuantizedIndex::buildClustering(int nClusters, int numThreads) {
    int n = ntotal.load();
    if (n == 0) return;

    if (nClusters <= 0) {
        // 默认: sqrt(N/d) 个聚类
        nClusters = std::max(1, (int)std::sqrt((double)n / d));
    }

    std::cout << "Building K-Means clustering with " << nClusters << " clusters for " << n << " vectors..." << std::endl;

    // 使用新的聚类策略模块
    KMeansClustering strategy(20, numThreads);
    ClusteringResult result = strategy.cluster(*this, nClusters);

    // 更新索引内部状态
    numClusters = result.numClusters;
    clusterAssignment = std::move(result.assignments);
    hasClustering = true;

    // 打印统计信息
    result.print();
}



int HNSWQuantizedIndex::getClusterAssignment(int nodeId) const {
    if (!hasClustering || nodeId < 0 || nodeId >= (int)clusterAssignment.size()) {
        return -1;
    }
    return clusterAssignment[nodeId];
}

// ========== 序列化 ==========

void HNSWQuantizedIndex::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for writing: " + filename);
    
    // 写入魔数和版本
    const char magic[] = "HNSW_Q8";
    int version = 4;  // v4: 增加子组信息
    ofs.write(magic, 7);
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // 写入基本参数
    int n = ntotal.load();
    int ep = entryPoint.load();
    int maxLev = currentMaxLevel.load();
    int distTypeInt = static_cast<int>(distType);

    ofs.write(reinterpret_cast<const char*>(&d), sizeof(d));
    ofs.write(reinterpret_cast<const char*>(&M), sizeof(M));
    ofs.write(reinterpret_cast<const char*>(&M0), sizeof(M0));
    ofs.write(reinterpret_cast<const char*>(&efConstruction), sizeof(efConstruction));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    ofs.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
    ofs.write(reinterpret_cast<const char*>(&maxLev), sizeof(maxLev));
    ofs.write(reinterpret_cast<const char*>(&distTypeInt), sizeof(distTypeInt));
    
    // 写入量化器参数
    quantizer.save(ofs);
    
    // 写入量化向量
    ofs.write(reinterpret_cast<const char*>(quantizedVectors.data()), (size_t)n * d);

    // 写入层数
    ofs.write(reinterpret_cast<const char*>(levels.data()), (size_t)n * sizeof(int));
    
    // 写入邻居
    for (int i = 0; i < n; ++i) {
        int numLevels = neighbors[i].size();
        ofs.write(reinterpret_cast<const char*>(&numLevels), sizeof(numLevels));
        for (int lev = 0; lev < numLevels; ++lev) {
            int numNbrs = neighbors[i][lev].size();
            ofs.write(reinterpret_cast<const char*>(&numNbrs), sizeof(numNbrs));
            if (numNbrs > 0) {
                ofs.write(reinterpret_cast<const char*>(neighbors[i][lev].data()), numNbrs * sizeof(int32_t));
            }
        }
    }
    
    // 写入聚类信息
    ofs.write(reinterpret_cast<const char*>(&hasClustering), sizeof(hasClustering));
    if (hasClustering) {
        ofs.write(reinterpret_cast<const char*>(&numClusters), sizeof(numClusters));
        ofs.write(reinterpret_cast<const char*>(clusterAssignment.data()), (size_t)n * sizeof(int));
    }

    // 写入子组信息（v4+）
    saveSubgroupInfo(ofs);
}

void HNSWQuantizedIndex::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file for reading: " + filename);
    
    // 读取魔数和版本
    char magic[8] = {0};
    int version;
    ifs.read(magic, 7);
    if (std::string(magic) != "HNSW_Q8") {
        throw std::runtime_error("Invalid file format");
    }
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // 读取基本参数
    int n, ep, maxLev;
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
    ifs.read(reinterpret_cast<char*>(&M), sizeof(M));
    ifs.read(reinterpret_cast<char*>(&M0), sizeof(M0));
    ifs.read(reinterpret_cast<char*>(&efConstruction), sizeof(efConstruction));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    ifs.read(reinterpret_cast<char*>(&ep), sizeof(ep));
    ifs.read(reinterpret_cast<char*>(&maxLev), sizeof(maxLev));

    // v3: 读取 distType
    if (version >= 3) {
        int distTypeInt;
        ifs.read(reinterpret_cast<char*>(&distTypeInt), sizeof(distTypeInt));
        distType = static_cast<DistanceType>(distTypeInt);
    }
    // v2 及更早版本：distType 保持构造时的值（不覆盖）

    ntotal.store(n);
    entryPoint.store(ep);
    currentMaxLevel.store(maxLev);

    // 读取量化器参数
    quantizer = ScalarQuantizer(d);
    quantizer.load(ifs);
    
    // 读取量化向量
    quantizedVectors.resize((size_t)n * d);
    ifs.read(reinterpret_cast<char*>(quantizedVectors.data()), (size_t)n * d);

    // 读取层数
    levels.resize(n);
    ifs.read(reinterpret_cast<char*>(levels.data()), (size_t)n * sizeof(int));
    
    // 读取邻居
    neighbors.resize(n);
    for (int i = 0; i < n; ++i) {
        int numLevels;
        ifs.read(reinterpret_cast<char*>(&numLevels), sizeof(numLevels));
        neighbors[i].resize(numLevels);
        for (int lev = 0; lev < numLevels; ++lev) {
            int numNbrs;
            ifs.read(reinterpret_cast<char*>(&numNbrs), sizeof(numNbrs));
            neighbors[i][lev].resize(numNbrs);
            if (numNbrs > 0) {
                ifs.read(reinterpret_cast<char*>(neighbors[i][lev].data()), numNbrs * sizeof(int32_t));
            }
        }
    }
    
    // 读取聚类信息（版本2+）
    if (version >= 2) {
        ifs.read(reinterpret_cast<char*>(&hasClustering), sizeof(hasClustering));
        if (hasClustering) {
            ifs.read(reinterpret_cast<char*>(&numClusters), sizeof(numClusters));
            clusterAssignment.resize(n);
            ifs.read(reinterpret_cast<char*>(clusterAssignment.data()), (size_t)n * sizeof(int));
        }
    }

    // 读取子组信息（版本4+）
    if (version >= 4) {
        loadSubgroupInfo(ifs);
    }
}


// ========== 图分区聚类 ==========

void HNSWQuantizedIndex::buildGraphPartitionClustering(int nClusters, double imbalance) {
    int n = ntotal.load();
    if (n == 0) return;

    if (nClusters <= 0) {
        // 默认: sqrt(N/d) 个聚类 (与K-Means一致，基于EmbeddingPIR维度优化)
        nClusters = std::max(1, (int)std::sqrt((double)n / d));
    }

    std::cout << "Building graph partition clustering with " << nClusters
              << " clusters for " << n << " vectors..." << std::endl;

    // 使用新的聚类策略模块
    GraphPartitionClustering strategy(imbalance);
    ClusteringResult result = strategy.cluster(*this, nClusters);

    // 更新索引内部状态
    numClusters = result.numClusters;
    clusterAssignment = std::move(result.assignments);
    hasClustering = true;

    // 打印统计信息
    result.print();
}

HNSWQuantizedIndex::ClusteringStats HNSWQuantizedIndex::analyzeCurrentClustering() const {
    ClusteringStats stats{};

    if (!hasClustering) {
        return stats;
    }

    int n = ntotal.load();
    stats.numClusters = numClusters;

    // Compute cluster sizes
    std::vector<int> clusterSizes(numClusters, 0);
    for (int i = 0; i < n; i++) {
        int c = clusterAssignment[i];
        if (c >= 0 && c < numClusters) {
            clusterSizes[c]++;
        }
    }

    stats.maxClusterSize = *std::max_element(clusterSizes.begin(), clusterSizes.end());
    stats.minClusterSize = *std::min_element(clusterSizes.begin(), clusterSizes.end());
    stats.avgClusterSize = static_cast<double>(n) / numClusters;
    stats.imbalance = (stats.maxClusterSize - stats.avgClusterSize) / stats.avgClusterSize;

    // Compute neighbor clusters and edge cut
    double totalNeighborClusters = 0;
    int totalEdges = 0;
    int edgeCut = 0;

    for (int u = 0; u < n; u++) {
        std::set<int> neighborClusters;
        if (!neighbors[u].empty()) {
            for (int32_t v : neighbors[u][0]) {
                if (v >= 0 && v < n) {
                    neighborClusters.insert(clusterAssignment[v]);
                    totalEdges++;
                    if (clusterAssignment[u] != clusterAssignment[v]) {
                        edgeCut++;
                    }
                }
            }
        }
        totalNeighborClusters += neighborClusters.size();
    }

    stats.avgNeighborClusters = totalNeighborClusters / n;
    stats.edgeCutRatio = (totalEdges > 0) ? static_cast<double>(edgeCut) / totalEdges : 0;

    return stats;
}

// ============================================================================
// 延迟量化模式：使用 Float 向量构建图
// ============================================================================

float HNSWQuantizedIndex::computeDistanceFloat(int32_t id, const float* query) const {
    const float* vec = floatVectors.data() + (size_t)id * d;
    float dist = 0;

    if (distType == DistanceType::L2) {
        for (int i = 0; i < d; ++i) {
            float diff = vec[i] - query[i];
            dist += diff * diff;
        }
    } else {
        // InnerProduct: 越大越好，取负数使其越小越好
        for (int i = 0; i < d; ++i) {
            dist -= vec[i] * query[i];
        }
    }
    return dist;
}

int32_t HNSWQuantizedIndex::greedySearchFloat(const float* query, int32_t entry, int level) const {
    int32_t curr = entry;
    float currDist = computeDistanceFloat(curr, query);

    bool changed = true;
    while (changed) {
        changed = false;
        if (curr < 0 || curr >= (int)neighbors.size()) break;
        if (level < 0 || level >= (int)neighbors[curr].size()) break;

        const auto& nbrs = neighbors[curr][level];
        for (int32_t nbr : nbrs) {
            if (nbr < 0 || nbr >= (int)neighbors.size()) continue;
            float d = computeDistanceFloat(nbr, query);
            if (d < currDist) {
                currDist = d;
                curr = nbr;
                changed = true;
            }
        }
    }
    return curr;
}

std::priority_queue<HNSWQuantizedIndex::NodeDistFloat,
                    std::vector<HNSWQuantizedIndex::NodeDistFloat>,
                    std::greater<HNSWQuantizedIndex::NodeDistFloat>>
HNSWQuantizedIndex::searchLayerFloat(const float* query, int32_t entry, int ef, int level, VisitedTable& vt) const {
    std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>> candidates;
    std::priority_queue<NodeDistFloat> results;

    float d = computeDistanceFloat(entry, query);
    candidates.push({d, entry});
    results.push({d, entry});
    vt.set(entry);

    while (!candidates.empty()) {
        NodeDistFloat curr = candidates.top();
        candidates.pop();

        if (curr.distance > results.top().distance) break;

        if (curr.id < 0 || curr.id >= (int)neighbors.size()) continue;
        if (level < 0 || level >= (int)neighbors[curr.id].size()) continue;

        for (int32_t nbr : neighbors[curr.id][level]) {
            if (nbr < 0 || nbr >= (int)neighbors.size()) continue;
            if (vt.get(nbr)) continue;
            vt.set(nbr);

            float nd = computeDistanceFloat(nbr, query);
            if (nd < results.top().distance || (int)results.size() < ef) {
                candidates.push({nd, nbr});
                results.push({nd, nbr});
                if ((int)results.size() > ef) results.pop();
            }
        }
    }

    std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>> ret;
    while (!results.empty()) {
        ret.push(results.top());
        results.pop();
    }
    return ret;
}

std::vector<int32_t> HNSWQuantizedIndex::selectNeighborsFloat(
    const float* query,
    std::priority_queue<NodeDistFloat, std::vector<NodeDistFloat>, std::greater<NodeDistFloat>>& candidates,
    int maxM) const {

    // 首先保存所有候选者到向量中（与量化版本一致）
    std::vector<NodeDistFloat> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }

    std::vector<int32_t> result;
    result.reserve(maxM);

    // 启发式邻居选择
    for (const auto& nd : sorted) {
        if ((int)result.size() >= maxM) break;

        bool good = true;
        for (int32_t sel : result) {
            float selDist = computeDistanceFloat(sel, floatVectors.data() + (size_t)nd.id * d);
            if (selDist < nd.distance) {
                good = false;
                break;
            }
        }
        if (good) result.push_back(nd.id);
    }

    // 如果启发式选择不够，补充最近的
    if ((int)result.size() < maxM) {
        for (const auto& nd : sorted) {
            if ((int)result.size() >= maxM) break;
            if (std::find(result.begin(), result.end(), nd.id) == result.end()) {
                result.push_back(nd.id);
            }
        }
    }

    return result;
}


void HNSWQuantizedIndex::addConnectionFloatThreadSafe(int32_t from, int32_t to, int level) {
    // 边界检查：确保 from 节点有这一层
    if (from < 0 || from >= (int)neighbors.size()) return;
    if (level < 0 || level >= (int)neighbors[from].size()) return;

    std::lock_guard<std::mutex> lock(nodeLocks[from]);
    auto& fromNbrs = neighbors[from][level];
    int maxConn = (level == 0) ? M0 : M;

    // 检查是否已经存在
    for (int32_t nbr : fromNbrs) {
        if (nbr == to) return;
    }

    if ((int)fromNbrs.size() < maxConn) {
        fromNbrs.push_back(to);
    } else {
        // 使用简单替换逻辑（线程安全版本避免复杂计算）
        const float* fromVec = floatVectors.data() + (size_t)from * d;
        int maxId = (int)(floatVectors.size() / d);
        if (from >= maxId || to >= maxId || to < 0) return;

        float toDist = computeDistanceFloat(to, fromVec);

        float worstDist = -1;
        int worstIdx = -1;
        for (int i = 0; i < (int)fromNbrs.size(); ++i) {
            int32_t nbr = fromNbrs[i];
            if (nbr >= 0 && nbr < maxId) {
                float dist = computeDistanceFloat(nbr, fromVec);
                if (dist > worstDist) {
                    worstDist = dist;
                    worstIdx = i;
                }
            }
        }

        if (worstIdx >= 0 && toDist < worstDist) {
            fromNbrs[worstIdx] = to;
        }
    }
}

void HNSWQuantizedIndex::addOneFloatParallel(int32_t id, int level, int totalNodes) {
    const float* query = floatVectors.data() + (size_t)id * d;

    int32_t ep = entryPoint.load();
    int maxLev = currentMaxLevel.load();

    for (int lev = maxLev; lev > level; --lev) {
        ep = greedySearchFloat(query, ep, lev);
    }

    VisitedTable vt(totalNodes);
    for (int lev = std::min(level, maxLev); lev >= 0; --lev) {
        vt.reset();
        auto candidates = searchLayerFloat(query, ep, efConstruction, lev, vt);

        int maxConn = (lev == 0) ? M0 : M;
        auto selected = selectNeighborsFloat(query, candidates, maxConn);

        neighbors[id][lev] = selected;

        for (int32_t sel : selected) {
            addConnectionFloatThreadSafe(sel, id, lev);
        }

        if (!selected.empty()) {
            ep = selected[0];
        }
    }

    if (level > maxLev) {
        std::lock_guard<std::mutex> lock(globalMutex);
        if (level > currentMaxLevel.load()) {
            currentMaxLevel.store(level);
            entryPoint.store(id);
        }
    }
}

void HNSWQuantizedIndex::addFloatParallel(const float* data, int n, int numThreads) {
    if (n <= 0) return;

#ifdef _OPENMP
    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }
    omp_set_num_threads(numThreads);
#else
    numThreads = 1;
#endif

    int startId = ntotal.load();
    useFloatBuild = true;

    // 预分配空间
    floatVectors.resize((size_t)(startId + n) * d);
    neighbors.resize(startId + n);
    levels.resize(startId + n);
    nodeLocks = std::vector<std::mutex>(startId + n);

    // 复制原始 float 向量（不量化）
    std::cout << "Copying " << n << " float vectors..." << std::endl;
    std::memcpy(floatVectors.data() + (size_t)startId * d, data, (size_t)n * d * sizeof(float));

    // 训练量化器（为后续 finalizeQuantization 做准备）
    if (startId == 0 && n > 0) {
        quantizer.train(data, n);
    }

    // 预先计算所有层数
    std::cout << "Generating levels..." << std::endl;
    for (int i = 0; i < n; ++i) {
        int id = startId + i;
        int level = randomLevel();
        levels[id] = level;
        neighbors[id].resize(level + 1);
    }

    // 第一个向量单独处理
    if (startId == 0) {
        entryPoint.store(0);
        currentMaxLevel.store(levels[0]);
    }

    ntotal.store(startId + n);

    // 批量并行插入
    std::cout << "Building index with " << numThreads << " threads (float mode)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    const int batchSize = std::max(1000, n / 100);
    std::atomic<int> processed(1);

    for (int batchStart = 1; batchStart < n; batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, n);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int i = batchStart; i < batchEnd; ++i) {
            int id = startId + i;
            addOneFloatParallel(id, levels[id], startId + n);

            int p = processed.fetch_add(1) + 1;
            if (p % 10000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - startTime).count();
#ifdef _OPENMP
                #pragma omp critical
#endif
                {
                    std::cout << "Added " << p << "/" << n << " vectors, "
                              << std::fixed << std::setprecision(1) << elapsed << "s elapsed, "
                              << p / elapsed << " vec/s" << std::endl;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "Float build completed: " << n << " vectors in "
              << std::fixed << std::setprecision(2) << totalTime << "s ("
              << n / totalTime << " vec/s)" << std::endl;
}

void HNSWQuantizedIndex::addFloatParallelMove(std::vector<float>&& data, int n, int numThreads) {
    if (n <= 0) return;

    int startId = ntotal.load();
    if (startId != 0) {
        // 非首次添加，回退到拷贝方式
        addFloatParallel(data.data(), n, numThreads);
        return;
    }

#ifdef _OPENMP
    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }
    omp_set_num_threads(numThreads);
#else
    numThreads = 1;
#endif

    useFloatBuild = true;

    // 直接接管输入 vector，零拷贝
    floatVectors = std::move(data);
    floatVectors.resize((size_t)n * d);  // 确保大小正确
    neighbors.resize(n);
    levels.resize(n);
    nodeLocks = std::vector<std::mutex>(n);

    std::cout << "Zero-copy: moved " << n << " float vectors directly." << std::endl;

    // 训练量化器
    quantizer.train(floatVectors.data(), n);

    // 预先计算所有层数
    std::cout << "Generating levels..." << std::endl;
    for (int i = 0; i < n; ++i) {
        int level = randomLevel();
        levels[i] = level;
        neighbors[i].resize(level + 1);
    }

    entryPoint.store(0);
    currentMaxLevel.store(levels[0]);
    ntotal.store(n);

    // 批量并行插入
    std::cout << "Building index with " << numThreads << " threads (float mode)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    const int batchSize = std::max(1000, n / 100);
    std::atomic<int> processed(1);

    for (int batchStart = 1; batchStart < n; batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, n);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int i = batchStart; i < batchEnd; ++i) {
            addOneFloatParallel(i, levels[i], n);

            int p = processed.fetch_add(1) + 1;
            if (p % 10000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - startTime).count();
#ifdef _OPENMP
                #pragma omp critical
#endif
                {
                    std::cout << "Added " << p << "/" << n << " vectors, "
                              << std::fixed << std::setprecision(1) << elapsed << "s elapsed, "
                              << p / elapsed << " vec/s" << std::endl;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "Float build completed: " << n << " vectors in "
              << std::fixed << std::setprecision(2) << totalTime << "s ("
              << n / totalTime << " vec/s)" << std::endl;
}

void HNSWQuantizedIndex::finalizeQuantization() {
    if (!useFloatBuild) {
        std::cout << "Warning: finalizeQuantization called but not in float build mode" << std::endl;
        return;
    }

    int n = ntotal.load();
    std::cout << "Finalizing quantization for " << n << " vectors..." << std::endl;

    // 分配量化向量空间
    quantizedVectors.resize((size_t)n * d);

    // 量化所有向量
    quantizer.quantize(floatVectors.data(), quantizedVectors.data(), n);

    // 释放 float 向量内存
    floatVectors.clear();
    floatVectors.shrink_to_fit();
    useFloatBuild = false;

    std::cout << "Quantization complete. Float vectors released." << std::endl;
}

// ============================================================================
// 子组相关方法实现
// ============================================================================

void HNSWQuantizedIndex::renumberNodesByCluster() {
    if (!hasClustering) {
        throw std::runtime_error("renumberNodesByCluster requires clustering data");
    }

    // 使用 SubgroupManager 执行重编号
    SubgroupManager mgr;
    mgr.renumberNodesByCluster(*this, clusterAssignment, numClusters);

    // Move 结果到内部数组，零拷贝
    nodeToNewId = mgr.moveNodeToNewId();
    newIdToNode = mgr.moveNewIdToNode();
    clusterOffset = mgr.moveClusterOffsets();
}

void HNSWQuantizedIndex::buildSubgroups(int targetSubgroupSize) {
    if (!hasClustering || nodeToNewId.empty()) {
        throw std::runtime_error("buildSubgroups requires clustering and node renumbering");
    }

    // 使用已有的重编号数据初始化 SubgroupManager（避免重复 BFS）
    SubgroupManager mgr;
    mgr.setRenumberData(nodeToNewId, newIdToNode, clusterOffset, numClusters);

    // 然后调用 buildSubgroups
    mgr.buildSubgroups(*this, clusterAssignment, targetSubgroupSize);

    // Move 结果到内部数组，零拷贝
    maxSubgroupSize = mgr.getMaxSubgroupSize();
    numSubgroupsPerCluster = mgr.getTotalSubgroups() / numClusters;
    subgroupOffset = mgr.moveSubgroupOffsets();
    nodeSubgroup = mgr.moveNodeSubgroup();
    nodeLocalIdxInSubgroup = mgr.moveNodeLocalIdx();
    hasSubgrouping = true;
}

void HNSWQuantizedIndex::buildNeighborInfo() {
    if (!hasSubgrouping) {
        throw std::runtime_error("buildNeighborInfo requires subgroup data");
    }

    // 使用 SubgroupManager 执行邻居信息构建
    SubgroupManager mgr;
    mgr.initFromExistingData(
        nodeToNewId, newIdToNode, clusterOffset, subgroupOffset,
        nodeSubgroup, nodeLocalIdxInSubgroup,
        std::vector<hnsw::NodeNeighborInfo>(),  // 空邻居信息
        numClusters, maxSubgroupSize
    );

    // 调用 SubgroupManager 的 buildNeighborInfo
    mgr.buildNeighborInfo(*this, clusterAssignment);

    // 类型统一后直接 move，零拷贝
    nodeNeighborInfo = mgr.moveAllNeighborInfo();
}





void HNSWQuantizedIndex::saveSubgroupInfo(std::ofstream& ofs) const {
    // 写入子组标志
    ofs.write(reinterpret_cast<const char*>(&hasSubgrouping), sizeof(hasSubgrouping));

    if (!hasSubgrouping) return;

    // 写入子组参数
    ofs.write(reinterpret_cast<const char*>(&numSubgroupsPerCluster), sizeof(numSubgroupsPerCluster));
    ofs.write(reinterpret_cast<const char*>(&maxSubgroupSize), sizeof(maxSubgroupSize));

    // 写入节点映射
    int n = ntotal.load();
    ofs.write(reinterpret_cast<const char*>(nodeToNewId.data()), (size_t)n * sizeof(int));
    ofs.write(reinterpret_cast<const char*>(newIdToNode.data()), (size_t)n * sizeof(int));

    // 写入节点子组映射 (nodeSubgroup[newId] -> subgroupId)
    ofs.write(reinterpret_cast<const char*>(nodeSubgroup.data()), (size_t)n * sizeof(int));

    // 写入节点在子组内的局部索引 (nodeLocalIdxInSubgroup[newId] -> posInSubgroup)
    ofs.write(reinterpret_cast<const char*>(nodeLocalIdxInSubgroup.data()), (size_t)n * sizeof(int));

    // 写入聚类偏移
    int numOffsets = (int)clusterOffset.size();
    ofs.write(reinterpret_cast<const char*>(&numOffsets), sizeof(numOffsets));
    ofs.write(reinterpret_cast<const char*>(clusterOffset.data()), numOffsets * sizeof(int));

    // 写入子组偏移
    for (int c = 0; c < numClusters; ++c) {
        int numSubOffsets = (int)subgroupOffset[c].size();
        ofs.write(reinterpret_cast<const char*>(&numSubOffsets), sizeof(numSubOffsets));
        ofs.write(reinterpret_cast<const char*>(subgroupOffset[c].data()), numSubOffsets * sizeof(int));
    }

    // 写入邻居信息
    for (int i = 0; i < n; ++i) {
        int numGroups = (int)nodeNeighborInfo[i].groups.size();
        ofs.write(reinterpret_cast<const char*>(&numGroups), sizeof(numGroups));

        for (const auto& group : nodeNeighborInfo[i].groups) {
            ofs.write(reinterpret_cast<const char*>(&group.cluster), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&group.subgroup), sizeof(int));
            int numIndices = (int)group.localIndices.size();
            ofs.write(reinterpret_cast<const char*>(&numIndices), sizeof(numIndices));
            if (numIndices > 0) {
                ofs.write(reinterpret_cast<const char*>(group.localIndices.data()), numIndices * sizeof(int));
            }
        }
    }

    std::cout << "Subgroup info saved." << std::endl;
}

void HNSWQuantizedIndex::loadSubgroupInfo(std::ifstream& ifs) {
    // 读取子组标志
    ifs.read(reinterpret_cast<char*>(&hasSubgrouping), sizeof(hasSubgrouping));

    if (!hasSubgrouping) return;

    int n = ntotal.load();

    // 读取子组参数
    ifs.read(reinterpret_cast<char*>(&numSubgroupsPerCluster), sizeof(numSubgroupsPerCluster));
    ifs.read(reinterpret_cast<char*>(&maxSubgroupSize), sizeof(maxSubgroupSize));

    // 读取节点映射
    nodeToNewId.resize(n);
    newIdToNode.resize(n);
    ifs.read(reinterpret_cast<char*>(nodeToNewId.data()), (size_t)n * sizeof(int));
    ifs.read(reinterpret_cast<char*>(newIdToNode.data()), (size_t)n * sizeof(int));

    // 读取节点子组映射
    nodeSubgroup.resize(n);
    ifs.read(reinterpret_cast<char*>(nodeSubgroup.data()), (size_t)n * sizeof(int));

    // 读取节点在子组内的局部索引
    nodeLocalIdxInSubgroup.resize(n);
    ifs.read(reinterpret_cast<char*>(nodeLocalIdxInSubgroup.data()), (size_t)n * sizeof(int));

    // 读取聚类偏移
    int numOffsets;
    ifs.read(reinterpret_cast<char*>(&numOffsets), sizeof(numOffsets));
    clusterOffset.resize(numOffsets);
    ifs.read(reinterpret_cast<char*>(clusterOffset.data()), numOffsets * sizeof(int));

    // 读取子组偏移
    subgroupOffset.resize(numClusters);
    for (int c = 0; c < numClusters; ++c) {
        int numSubOffsets;
        ifs.read(reinterpret_cast<char*>(&numSubOffsets), sizeof(numSubOffsets));
        subgroupOffset[c].resize(numSubOffsets);
        ifs.read(reinterpret_cast<char*>(subgroupOffset[c].data()), numSubOffsets * sizeof(int));
    }

    // 读取邻居信息
    nodeNeighborInfo.resize(n);
    for (int i = 0; i < n; ++i) {
        int numGroups;
        ifs.read(reinterpret_cast<char*>(&numGroups), sizeof(numGroups));
        nodeNeighborInfo[i].groups.resize(numGroups);

        for (int g = 0; g < numGroups; ++g) {
            ifs.read(reinterpret_cast<char*>(&nodeNeighborInfo[i].groups[g].cluster), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&nodeNeighborInfo[i].groups[g].subgroup), sizeof(int));
            int numIndices;
            ifs.read(reinterpret_cast<char*>(&numIndices), sizeof(numIndices));
            nodeNeighborInfo[i].groups[g].localIndices.resize(numIndices);
            if (numIndices > 0) {
                ifs.read(reinterpret_cast<char*>(nodeNeighborInfo[i].groups[g].localIndices.data()), numIndices * sizeof(int));
            }
        }
    }

    std::cout << "Subgroup info loaded." << std::endl;
}

// ============================================================================
// 获取 SubgroupManager（从现有数据构建）
// ============================================================================
SubgroupManager HNSWQuantizedIndex::getSubgroupManager() const {
    SubgroupManager mgr;

    if (!hasSubgrouping) {
        return mgr;  // 返回空的 SubgroupManager
    }

    // 类型统一后无需转换，直接传引用
    mgr.initFromExistingData(
        nodeToNewId,
        newIdToNode,
        clusterOffset,
        subgroupOffset,
        nodeSubgroup,
        nodeLocalIdxInSubgroup,
        nodeNeighborInfo,
        numClusters,
        maxSubgroupSize
    );

    return mgr;
}

SubgroupManager HNSWQuantizedIndex::moveToSubgroupManager() {
    SubgroupManager mgr;

    if (!hasSubgrouping) {
        return mgr;
    }

    // Move 版本：零拷贝转移所有数据
    mgr.initFromExistingDataMove(
        std::move(nodeToNewId),
        std::move(newIdToNode),
        std::move(clusterOffset),
        std::move(subgroupOffset),
        std::move(nodeSubgroup),
        std::move(nodeLocalIdxInSubgroup),
        std::move(nodeNeighborInfo),
        numClusters,
        maxSubgroupSize
    );

    return mgr;
}

// ============================================================================
// hnswlib 加速构建
// ============================================================================

#ifdef USE_HNSWLIB

void HNSWQuantizedIndex::buildWithHnswlib(const float* data, int n, int numThreads) {
    // 拷贝数据后调用 move 版本
    std::vector<float> dataCopy(data, data + (size_t)n * d);
    buildWithHnswlibMove(std::move(dataCopy), n, numThreads);
}

void HNSWQuantizedIndex::buildWithHnswlibMove(std::vector<float>&& data, int n, int numThreads) {
    if (n <= 0) return;

#ifdef _OPENMP
    if (numThreads <= 0) {
        numThreads = omp_get_max_threads();
    }
    omp_set_num_threads(numThreads);
#else
    numThreads = 1;
#endif

    // 1. 训练量化器
    quantizer.train(data.data(), n);

    // 2. 创建 hnswlib 索引
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (distType == DistanceType::L2) {
        space = new hnswlib::L2Space(d);
    } else {
        space = new hnswlib::InnerProductSpace(d);
    }

    std::cout << "Building HNSW graph with hnswlib (n=" << n << ", M=" << M
              << ", efConstruction=" << efConstruction << ", threads=" << numThreads << ")..." << std::endl;

    hnswlib::HierarchicalNSW<float> hnsw(space, n, M, efConstruction);

    // 3. 添加第一个点（单线程），其余并行
    auto startTime = std::chrono::high_resolution_clock::now();

    hnsw.addPoint(data.data(), 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) num_threads(numThreads)
#endif
    for (int i = 1; i < n; ++i) {
        hnsw.addPoint(data.data() + (size_t)i * d, (size_t)i);

        if (i % 100000 == 0) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - startTime).count();
                std::cout << "  hnswlib: added " << i << "/" << n << " vectors, "
                          << std::fixed << std::setprecision(1) << elapsed << "s, "
                          << i / elapsed << " vec/s" << std::endl;
            }
        }
    }

    auto buildEnd = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double>(buildEnd - startTime).count();
    std::cout << "hnswlib graph build completed: " << n << " vectors in "
              << std::fixed << std::setprecision(2) << buildTime << "s ("
              << n / buildTime << " vec/s)" << std::endl;

    // 4. 提取图结构
    extractGraphFromHnswlib(&hnsw, n);

    // 5. 构建 internal_id -> label 映射
    std::vector<int> internalToLabel(n);
    for (int i = 0; i < n; ++i) {
        internalToLabel[i] = static_cast<int>(hnsw.getExternalLabel(i));
    }

    // 6. 从 hnswlib 内部存储量化（按 label 存放）
    std::cout << "Quantizing vectors from hnswlib internal storage..." << std::endl;
    quantizedVectors.resize((size_t)n * d);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(numThreads)
#endif
    for (int i = 0; i < n; ++i) {
        int label = internalToLabel[i];
        const float* vec = (const float*)hnsw.getDataByInternalId(i);
        quantizer.quantize(vec, quantizedVectors.data() + (size_t)label * d, 1);
    }

    // 7. 释放原始数据
    { std::vector<float>().swap(data); }

    ntotal.store(n);
    useFloatBuild = false;  // 已完成量化

    delete space;

    std::cout << "hnswlib build + quantization complete." << std::endl;
}

void HNSWQuantizedIndex::extractGraphFromHnswlib(void* hnswPtr, int n) {
    auto& hnsw = *static_cast<hnswlib::HierarchicalNSW<float>*>(hnswPtr);

    std::cout << "Extracting graph structure from hnswlib..." << std::endl;

    // 构建 internal_id -> label 映射（并行插入时 internal_id != label）
    std::vector<int> i2l(n);
    for (int i = 0; i < n; ++i) {
        i2l[i] = static_cast<int>(hnsw.getExternalLabel(i));
    }

    neighbors.resize(n);
    levels.resize(n);
    nodeLocks = std::vector<std::mutex>(n);

    // entryPoint 和 maxLevel 需要映射到 label 空间
    entryPoint.store(i2l[hnsw.enterpoint_node_]);
    currentMaxLevel.store(hnsw.maxlevel_);

    for (int i = 0; i < n; ++i) {
        int label = i2l[i];
        int nodeLevel = hnsw.element_levels_[i];
        levels[label] = nodeLevel;
        neighbors[label].resize(nodeLevel + 1);

        // Level-0: [count, neighbor0, neighbor1, ...]
        unsigned int* ll0 = (unsigned int*)hnsw.get_linklist0(i);
        unsigned int cnt0 = *ll0;
        neighbors[label][0].resize(cnt0);
        for (unsigned int j = 0; j < cnt0; ++j) {
            neighbors[label][0][j] = i2l[*(ll0 + 1 + j)];
        }

        // Higher levels
        for (int lev = 1; lev <= nodeLevel; ++lev) {
            unsigned int* ll = (unsigned int*)hnsw.get_linklist(i, lev);
            unsigned int cnt = *ll;
            neighbors[label][lev].resize(cnt);
            for (unsigned int j = 0; j < cnt; ++j) {
                neighbors[label][lev][j] = i2l[*(ll + 1 + j)];
            }
        }
    }

    std::cout << "Graph extracted: " << n << " nodes, maxLevel=" << hnsw.maxlevel_
              << ", entryPoint=" << entryPoint.load() << std::endl;
}

#endif // USE_HNSWLIB

} // namespace hnsw

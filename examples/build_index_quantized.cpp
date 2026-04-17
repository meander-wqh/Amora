/**
 * @file build_index_quantized.cpp
 * @brief 构建量化 HNSW 索引 (支持多线程并行构建和聚类)
 *
 * 功能：
 * 1. 读取 fvecs/bvecs 格式的向量数据
 * 2. 将 Float32 向量量化为 Uint8
 * 3. 构建量化 HNSW 索引（并行构建）
 * 4. 对量化向量进行 K-Means 聚类
 * 5. 保存索引为 .hnsw_q8 格式（聚类数据嵌入索引文件中）
 *
 * 用法：
 *   ./build_index_quantized <base.fvecs> <output.hnsw_q8> [M] [efConstruction] [scale] [threads] [nClusters]
 */

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

/**
 * @brief 分析数据范围，自动计算最佳 scale 参数
 */
float analyzeDataRange(const float* data, int n, int dim, bool* hasNegative = nullptr) {
    float minVal = data[0], maxVal = data[0];
    
    // 采样分析（避免遍历全部数据）
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

    // 计算 scale：将数据映射到 [0, 255]
    float absMax = std::max(std::abs(minVal), std::abs(maxVal));
    float scale;
    
    if (minVal >= 0) {
        // 非负数据（如 SIFT）: 映射到 [0, 255]
        scale = 255.0f / maxVal;
        std::cout << "Data type: non-negative, using unsigned quantization" << std::endl;
    } else {
        // 有正负值的数据: 映射到 [-128, 127] 或 [0, 255]
        scale = 127.0f / absMax;
        std::cout << "Data type: signed, consider using signed quantization" << std::endl;
    }
    
    std::cout << "Recommended scale: " << scale << std::endl;
    return scale;
}

int main(int argc, char** argv) {
    // 参数解析
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
    float userScale = (argc > 5) ? std::stof(argv[5]) : 0.0f;  // 0 表示自动计算
    int numThreads = (argc > 6) ? std::stoi(argv[6]) : 0;  // 0 表示自动
    int nClusters = (argc > 7) ? std::stoi(argv[7]) : -1;  // -1 表示自动 (sqrt(n))
    bool useGraphPartition = (argc > 8) ? (std::stoi(argv[8]) != 0) : true;  // 默认使用图分区
    bool normalize = true; // 归一化后量化，使内积等价于L2距离
    
    // ========================================
    // 1. 加载数据
    // ========================================
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
    
    // ========================================
    // 2. 分析数据并配置量化参数
    // ========================================
    std::cout << "\n[2] Analyzing data range..." << std::endl;
    
    bool hasNegative = false;
    float scale = userScale;
    if (scale <= 0) {
        scale = analyzeDataRange(base.data(), n, dim, &hasNegative);
    } else {
        // 用户指定 scale 时也检测符号
        hasNegative = *std::min_element(base.begin(), base.end()) < 0;
        std::cout << "Using user-specified scale: " << scale << std::endl;
    }

    // 配置量化器
    QuantizerConfig cfg;
    cfg.bits = 8;           // 8-bit 量化
    cfg.scale = scale;      // 缩放因子
    cfg.isUnsigned = !hasNegative;  // 自动检测：有负值时用有符号量化
    cfg.normalize = normalize;  // 是否在量化前归一化
    
    std::cout << "\nQuantization config:" << std::endl;
    std::cout << "  Bits: " << cfg.bits << std::endl;
    std::cout << "  Scale: " << cfg.scale << std::endl;
    std::cout << "  Mode: " << (cfg.isUnsigned ? "unsigned [0,255]" : "signed [-128,127]") << std::endl;
    std::cout << "  Normalize: " << (cfg.normalize ? "yes" : "no") << std::endl;
    
    // ========================================
    // 断点恢复: 检查 checkpoint 文件
    // ========================================
    std::string ckptGraph = indexFile + ".ckpt_graph";    // 图+量化完成
    std::string ckptCluster = indexFile + ".ckpt_cluster"; // 聚类+子组完成

    HNSWQuantizedIndex index(dim, M, efConstruction, cfg, DistanceType::L2);

    // 检查是否可以从断点恢复
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
        // ========================================
        // 3. 使用 Float 向量构建 HNSW 图（保留原始精度）
        // ========================================
        std::cout << "\n[3] Building HNSW graph..." << std::endl;
        std::cout << "Parameters: M=" << M << ", efConstruction=" << efConstruction
                  << ", threads=" << (numThreads > 0 ? std::to_string(numThreads) : "auto") << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();

#ifdef USE_HNSWLIB
        // hnswlib 路径：图构建 + 量化一步完成
        std::cout << "Using hnswlib for fast graph building..." << std::endl;
        index.buildWithHnswlibMove(std::move(base), n, numThreads);
        // 不需要 finalizeQuantization()
#else
        // 原始路径：使用 float 向量构建图（零拷贝 move）
        std::cout << "Using built-in graph builder (float mode)..." << std::endl;
        index.addFloatParallelMove(std::move(base), n, numThreads);
        // 提前量化（释放 floatVectors，为图分区腾出内存）
        std::cout << "\n[3.5] Early quantization (free float vectors before clustering)..." << std::endl;
        index.finalizeQuantization();
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        buildTime = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\nGraph build completed!" << std::endl;
        std::cout << "  Build time: " << buildTime << "s" << std::endl;
        std::cout << "  Throughput: " << n / buildTime << " vectors/s" << std::endl;

        // 保存图断点（可通过环境变量 SKIP_CHECKPOINT=1 跳过）
        if (!std::getenv("SKIP_CHECKPOINT")) {
            std::cout << "  Saving graph checkpoint: " << ckptGraph << std::endl;
            index.save(ckptGraph);
            std::cout << "  Graph checkpoint saved." << std::endl;
        } else {
            std::cout << "  Skipping graph checkpoint (SKIP_CHECKPOINT=1)." << std::endl;
        }
    }

    if (!resumedFromCluster) {
        // ========================================
        // 4. 构建聚类 + 子组（计时）
        // ========================================
        auto tPost0 = std::chrono::high_resolution_clock::now();
        std::cout << "\n[4] Building clustering..." << std::endl;

        if (useGraphPartition) {
            std::cout << "Using graph partitioning (better balance)..." << std::endl;
            index.buildGraphPartitionClustering(nClusters, 0.05);  // 5% imbalance
        } else {
            std::cout << "Using K-means clustering..." << std::endl;
            index.buildClustering(nClusters);
        }

        // ========================================
        // 4.5 构建子组划分
        // ========================================
        std::cout << "\n[4.5] Building subgroup partitioning..." << std::endl;

        // 步骤1：按聚类顺序重新编号节点
        index.renumberNodesByCluster();

        // 步骤2：构建子组划分（顺序切分，BFS 重编号保证局部性）
        // 动态 targetSubgroupSize: 通信量最优公式
        int M0_val = 2 * M;  // 最大邻居数
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
        int numPartsCalc = (totalBitsCalc + 6) / 7;  // ceil(totalBits / 7)
        int targetSubgroupSize = std::max(120, (int)std::sqrt((double)n / (M0_val * numPartsCalc)));
        std::cout << "  Dynamic targetSubgroupSize: " << targetSubgroupSize
                  << " (totalBits=" << totalBitsCalc << ", numParts=" << numPartsCalc << ")" << std::endl;
        index.buildSubgroups(targetSubgroupSize);

        // 步骤3：生成邻居信息
        index.buildNeighborInfo();

        auto tPost1 = std::chrono::high_resolution_clock::now();
        double postTime = std::chrono::duration<double>(tPost1 - tPost0).count();
        std::cout << "\n  Clustering + subgroup time: " << postTime << "s" << std::endl;
        if (!resumedFromGraph) {
            std::cout << "  Graph build total: " << (buildTime + postTime) << "s"
                      << " (HNSW=" << buildTime << "s, post=" << postTime << "s)" << std::endl;
        }

        // 保存聚类断点（可通过环境变量 SKIP_CHECKPOINT=1 跳过）
        if (!std::getenv("SKIP_CHECKPOINT")) {
            std::cout << "  Saving cluster checkpoint: " << ckptCluster << std::endl;
            index.save(ckptCluster);
            std::cout << "  Cluster checkpoint saved." << std::endl;
        } else {
            std::cout << "  Skipping cluster checkpoint (SKIP_CHECKPOINT=1)." << std::endl;
        }
    }

    // ========================================
    // 6. 保存最终索引
    // ========================================
    std::cout << "\n[6] Saving final index to: " << indexFile << std::endl;

    index.save(indexFile);

    // 验证保存成功后再清理断点文件
    std::ifstream checkFile(indexFile, std::ios::binary | std::ios::ate);
    size_t savedSize = checkFile.tellg();
    size_t expectedMinSize = (size_t)n * dim + 1024;  // 至少 quantizedVectors + header
    checkFile.close();
    if (savedSize > expectedMinSize) {
        std::remove(ckptGraph.c_str());
        std::remove(ckptCluster.c_str());
        std::cout << "  Index saved (" << savedSize / 1024.0 / 1024.0 << " MB). Checkpoints cleaned up." << std::endl;
    } else {
        std::cerr << "  WARNING: Saved index too small (" << savedSize << " bytes), keeping checkpoints!" << std::endl;
    }
    
    // ========================================
    // 7. 输出统计信息
    // ========================================
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

    // 输出聚类质量统计
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

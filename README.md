# Private HNSW

基于 SimplePIR 协议的**隐私保护 HNSW 近似最近邻搜索**系统。服务端无法得知客户端查询了哪些数据，同时保持高搜索质量。

## 核心架构

采用**双 PIR（Private Information Retrieval）架构**，将搜索中的两类信息访问封装为独立的 PIR 查询：

| PIR 模块 | 功能 | 隐私保护 |
|----------|------|---------|
| **EmbeddingPIR** | 查询聚类内所有节点与 query 的距离 | 隐藏"查哪个聚类" |
| **NeighborPIR** | 查询子组内所有节点的邻居信息 | 隐藏"查哪个子组" |

**核心洞察**：SimplePIR 的矩阵乘法结构使得一次查询**免费获得整个聚类**（~11K 节点）的距离信息，结合图分区聚类的局部性，搜索中信息被大量复用。

## 编译

**依赖**：C++17, OpenMP, AVX2（可选，自动检测）

```bash
mkdir -p build && cd build
cmake .. && make -j16
```

## 使用

### 1. 下载数据集

```bash
# SIFT10K（小规模测试，~1.3 MB）
bash download_data_scripts/download_sift10K.sh

# SIFT1M（标准测试，~132 MB）
bash download_data_scripts/download_sift1M.sh

# SIFT10M（大规模测试，~13 GB）
bash download_data_scripts/download_sift10M.sh
```

### 2. 构建索引

```bash
./build/build_index_quantized <base.fvecs> <output.hnsw_q8> [M] [efConstruction] [scale] [threads] [nClusters] [graphPartition]
```

**示例**：

```bash
# SIFT1M（16线程，图分区聚类）
./build/build_index_quantized data/SIFT1M/sift_base.fvecs \
    data/index/sift1m_subgroup.hnsw_q8 16 200 0 16 -1 1

# SIFT10M（16线程）
./build/build_index_quantized data/SIFT10M/sift10m_base.bvecs \
    data/index/sift10m_subgroup.hnsw_q8 16 200 0 16 -1 1
```

**参数**：

| 参数 | 位置 | 说明 | 默认值 |
|------|------|------|--------|
| M | 3 | 每层邻居数 | 16 |
| efConstruction | 4 | 建图搜索束宽 | 200 |
| scale | 5 | 量化缩放因子 | 0（自动） |
| threads | 6 | 并行线程数 | 0（自动） |
| nClusters | 7 | 聚类数量 | -1（sqrt(N/d)） |
| graphPartition | 8 | 使用图分区聚类 | 1（是） |

### 3. 隐私搜索

```bash
./build/private_search_v2 <index> <queries> <groundtruth> <stats_output> [nq] [k] [ef] [ablation]
```

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| nq | 查询数量 | 全部 |
| k | 返回最近邻数 | 10 |
| ef | 搜索束宽 | 50 |
| ablation | 消融实验开关 | 无 |

**ablation 选项**（支持逗号分隔组合，如 `rounds_6,no_batch`）：

| 值 | 说明 |
|----|------|
| `no_prune` | 关闭质心剪枝 |
| `no_topcand` | 设置 topCand=0（不限制每聚类候选数） |
| `no_batch` | 关闭批量 PIR |
| `centroid_entry_K` | 覆盖入口聚类数为 K（如 `centroid_entry_3`） |
| `stale_N` | 连续 N 轮 top-k 无改善则早停（`stale_0` 关闭早停） |
| `rounds_N` | 覆盖 fixedSearchIterations 为 N（如 `rounds_6`） |

## 项目结构

```
private_hnsw/
├── include/                       # 头文件
│   ├── private_hnsw_v2.h          # 隐私搜索核心类 + 类型别名
│   ├── hnsw_quantized.h           # 量化 HNSW 索引
│   ├── graph_partitioner.h        # 图分区算法
│   ├── subgroup/subgroup_manager.h
│   └── clustering/clustering.h
├── src/                           # 实现
│   ├── private_hnsw_v2.cpp        # 隐私搜索核心逻辑
│   ├── hnsw_quantized.cpp
│   └── ...
├── simplepir/                     # PIR 协议层
│   ├── embedding_pir.h/cpp        # EmbeddingPIR（模板化）
│   ├── neighbor_pir.h/cpp         # NeighborPIR（模板化）
│   ├── pir_math.h/cpp             # AVX2 优化矩阵运算
│   └── ...
├── examples/                      # 可执行程序
│   ├── build_index_quantized.cpp  # 构建索引
│   └── private_search_v2.cpp      # 隐私搜索
├── docs/
│   └── Technique.md               # 详细技术文档
├── stats/                         # 搜索统计输出
└── data/                          # 数据集与索引
    ├── SIFT1M/
    ├── SIFT10M/
    └── index/
```

## 关键技术

| 技术 | 效果 |
|------|------|
| 固定通信轮次 | 每查询 PIR 次数恒定（dummy padding），通信模式不泄露隐私 |
| SimplePIR × 聚类协同 | 一次 EmbPIR 获得 ~11K 节点距离，搜索中大量复用 |
| 双 PIR 对称架构 | EmbeddingPIR + NeighborPIR 独立优化 |
| 批量 EmbPIR / NbrPIR | N 次查询合并为 1 次矩阵乘法，通信轮次 ~2-3x 压缩 |
| 压缩存储（uint32→uint8） | PIR 服务端内存带宽 ~4x 节省 |
| 图分区聚类 + BFS 重编号 | 跨聚类边 ~10%，子组局部性高 |
| 拆分编码 | 20+ 位邻居 ID 编码为多个 7-bit 部分 |
| 自适应质心剪枝 | 明文过滤 30-50% 候选聚类，零隐私泄露 |

详细技术说明参见 [Technique.md](docs/Technique.md)。

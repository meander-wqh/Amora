#!/bin/bash
# ============================================================
# download_deep1M.sh
# 下载 Deep1M (96维 CNN 深度特征) 数据集
# 来源: Yandex Deep-1B benchmark (via ann-benchmarks)
#
# 输出文件:
#   deep1m_base.fvecs        — 1,000,000 x 96 基础向量
#   deep1m_query.fvecs       — 10,000 x 96 查询向量
#   deep1m_groundtruth.ivecs — 10,000 x 100 真实最近邻 (L2)
#
# 依赖: wget, python3, numpy, h5py
# ============================================================

DATA_DIR="../data/Deep1M"
HDF5_URL="http://ann-benchmarks.com/deep-image-96-angular.hdf5"
HDF5_FILE="deep-image-96-angular.hdf5"

NUM_BASE=1000000
NUM_QUERY=10000
K_GT=100
DIM=96

mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

echo "============================================="
echo "  Deep1M (96-dim, L2) Dataset Downloader"
echo "============================================="

# ----------------------------------------------------------
# 0. 依赖检查
# ----------------------------------------------------------
echo "[0/4] 检查依赖..."
python3 -c "import numpy; import h5py" 2>/dev/null || {
    echo "  缺少 Python 依赖，正在安装 numpy 和 h5py..."
    pip install numpy h5py --break-system-packages -q
}

# ===========================================================
# 1. 下载 HDF5
# ===========================================================
if [ -f "deep1m_base.fvecs" ] && [ -f "deep1m_query.fvecs" ] && [ -f "deep1m_groundtruth.ivecs" ]; then
    echo "[1/4] 所有输出文件已存在，跳过下载"
elif [ -f "$HDF5_FILE" ]; then
    echo "[1/4] HDF5 文件已存在，跳过下载"
else
    echo "[1/4] 从 ann-benchmarks.com 下载 HDF5..."
    wget -c --timeout=120 --tries=3 "$HDF5_URL" -O "$HDF5_FILE"

    if [ ! -f "$HDF5_FILE" ] || [ $(stat -c%s "$HDF5_FILE" 2>/dev/null || stat -f%z "$HDF5_FILE" 2>/dev/null || echo 0) -lt 1048576 ]; then
        echo "  下载失败，请手动下载:"
        echo "  wget $HDF5_URL"
        echo "  放到: $(pwd)/"
        exit 1
    fi
    echo "  下载完成"
fi

# ===========================================================
# 2. HDF5 → fvecs/ivecs
# ===========================================================
if [ -f "deep1m_base.fvecs" ] && [ -f "deep1m_query.fvecs" ] && [ -f "deep1m_groundtruth.ivecs" ]; then
    echo "[2/4] 所有输出文件已存在，跳过处理"
else
    echo "[2/4] 解析 HDF5 并转换为 fvecs/ivecs..."

python3 << 'PYEOF'
import numpy as np
import h5py
import struct

NUM_BASE  = 1_000_000
NUM_QUERY = 10_000
K_GT      = 100

def write_fvecs(filename, data):
    """写入 fvecs 格式: 每行 [dim(int32), vec(float32 x dim)]"""
    n, d = data.shape
    data = data.astype(np.float32)
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(data[i].tobytes())
    print(f"  写入 {filename}: {n} x {d}")

def write_ivecs(filename, data):
    """写入 ivecs 格式: 每行 [k(int32), indices(int32 x k)]"""
    n, k = data.shape
    data = data.astype(np.int32)
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', k))
            f.write(data[i].tobytes())
    print(f"  写入 {filename}: {n} x {k}")

# ---- 加载 HDF5 ----
print("  加载 HDF5 文件...")
with h5py.File("deep-image-96-angular.hdf5", 'r') as f:
    print(f"  HDF5 keys: {list(f.keys())}")

    # train = base vectors, test = query vectors
    train = np.array(f['train'], dtype=np.float32)
    test  = np.array(f['test'],  dtype=np.float32)
    print(f"  train (base):  {train.shape}")
    print(f"  test  (query): {test.shape}")

    # neighbors 和 distances 是基于 angular 距离的 ground truth
    # 我们需要用 L2 重新计算 ground truth
    if 'neighbors' in f:
        ann_neighbors = np.array(f['neighbors'], dtype=np.int32)
        print(f"  neighbors:     {ann_neighbors.shape}")
    if 'distances' in f:
        ann_distances = np.array(f['distances'], dtype=np.float32)
        print(f"  distances:     {ann_distances.shape}")

# ---- 截取数据 ----
if train.shape[0] >= NUM_BASE:
    base_vectors = train[:NUM_BASE]
else:
    print(f"  注意: train 只有 {train.shape[0]} 条，全部使用")
    base_vectors = train
    NUM_BASE = train.shape[0]

if test.shape[0] >= NUM_QUERY:
    query_vectors = test[:NUM_QUERY]
else:
    query_vectors = test
    NUM_QUERY = test.shape[0]

print(f"  使用 base: {base_vectors.shape}, query: {query_vectors.shape}")

# ---- 写入 fvecs ----
print("  转换为 fvecs 格式...")
write_fvecs("deep1m_base.fvecs", base_vectors)
write_fvecs("deep1m_query.fvecs", query_vectors)

# ---- 计算 L2 Ground Truth ----
# ann-benchmarks 的 deep-image-96 用 angular 距离，我们需要 L2
print(f"  计算 L2 ground truth (k={K_GT})...")

gt_indices = np.zeros((NUM_QUERY, K_GT), dtype=np.int32)

BATCH = 1000
for start in range(0, NUM_QUERY, BATCH):
    end = min(start + BATCH, NUM_QUERY)
    q_batch = query_vectors[start:end]  # [batch, dim]

    # 分块计算 L2 距离，避免内存爆炸
    CHUNK = 200000
    l2_dist = np.zeros((end - start, NUM_BASE), dtype=np.float32)
    for c_start in range(0, NUM_BASE, CHUNK):
        c_end = min(c_start + CHUNK, NUM_BASE)
        # ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q·b
        diff = q_batch[:, np.newaxis, :] - base_vectors[np.newaxis, c_start:c_end, :]
        l2_dist[:, c_start:c_end] = np.sum(diff ** 2, axis=2)

    # 取 top-K 最近邻
    gt_indices[start:end] = np.argpartition(l2_dist, K_GT, axis=1)[:, :K_GT]

    # 对 top-K 内部排序
    for i in range(end - start):
        topk_idx = gt_indices[start + i]
        sorted_order = np.argsort(l2_dist[i, topk_idx])
        gt_indices[start + i] = topk_idx[sorted_order]

    print(f"    进度: {end}/{NUM_QUERY}")

write_ivecs("deep1m_groundtruth.ivecs", gt_indices)
print("\n  处理完成!")
PYEOF
fi

# ===========================================================
# 3. 清理
# ===========================================================
if [ -f "deep1m_base.fvecs" ] && [ -f "deep1m_query.fvecs" ] && [ -f "deep1m_groundtruth.ivecs" ]; then
    if [ -f "$HDF5_FILE" ]; then
        echo "[3/4] 清理中间文件..."
        read -p "是否删除 HDF5 文件以节省空间? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$HDF5_FILE"
            echo "  已清理"
        else
            echo "  保留 HDF5 文件"
        fi
    else
        echo "[3/4] 无中间文件需要清理"
    fi
else
    echo "[3/4] 跳过清理 (输出文件不完整，请保留 HDF5 以便重试)"
fi

# ===========================================================
# 4. 验证
# ===========================================================
echo "[4/4] 验证文件..."
echo "---------------------------------------------"

python3 << 'PYEOF'
import struct, os

def read_vecs_info(fname):
    size = os.path.getsize(fname)
    with open(fname, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
    record_size = 4 + dim * 4
    return size // record_size, dim

for f in ["deep1m_base.fvecs", "deep1m_query.fvecs", "deep1m_groundtruth.ivecs"]:
    if os.path.exists(f):
        n, d = read_vecs_info(f)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  OK {f:35s}  {n:>10,} x {d:<5d}  ({size_mb:,.1f} MB)")
    else:
        print(f"  FAIL {f:35s}  文件不存在")
PYEOF

echo "---------------------------------------------"
echo ""
echo "=== 完成! ==="
echo ""
echo "数据集信息:"
echo "  名称:     Deep1M (Yandex Deep-1B subset)"
echo "  维度:     96 (CNN 深度特征)"
echo "  基础向量: 1,000,000"
echo "  查询向量: 10,000"
echo "  GT (k):   100"
echo "  度量:     L2 (欧氏距离)"
echo "  格式:     fvecs / ivecs"
echo ""
echo "使用示例:"
echo "  ./build_index_quantized ../data/Deep1M/deep1m_base.fvecs \\"
echo "      ../data/index/deep1m_subgroup.hnsw_q8 \\"
echo "      -c graphpartition --subgroup"

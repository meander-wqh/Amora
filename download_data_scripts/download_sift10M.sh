#!/bin/bash
DATA_DIR="../data/SIFT10M"
mkdir -p $DATA_DIR && cd $DATA_DIR

echo "=== Downloading SIFT10M Dataset (from BIGANN/ANN_SIFT1B) ==="

# 1. 基础向量：流式解压 bigann_base.bvecs.gz，仅提取前 10M 向量
# bvecs 格式: 每向量 4+128=132 字节, 10M 向量 = 1.32 GB
if [ ! -f "sift10m_base.bvecs" ]; then
    echo "Downloading base vectors (streaming first 10M from BIGANN)..."
    wget -O - ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz \
        | gunzip -c \
        | head -c $((10000000 * 132)) \
        > sift10m_base.bvecs
    echo "Base vectors: $(wc -c < sift10m_base.bvecs) bytes (expect 1320000000)"
fi

# 2. 查询向量
if [ ! -f "sift10m_query.bvecs" ]; then
    echo "Downloading query vectors..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
    gunzip bigann_query.bvecs.gz
    mv bigann_query.bvecs sift10m_query.bvecs
fi

# 3. Ground truth (10M 子集)
if [ ! -f "sift10m_groundtruth.ivecs" ]; then
    echo "Downloading ground truth..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
    tar -xzf bigann_gnd.tar.gz
    mv gnd/idx_10M.ivecs sift10m_groundtruth.ivecs
    rm -rf gnd bigann_gnd.tar.gz
fi

ls -lh *.bvecs *.ivecs 2>/dev/null

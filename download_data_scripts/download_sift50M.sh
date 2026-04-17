#!/bin/bash
DATA_DIR="../data/SIFT50M"
mkdir -p $DATA_DIR && cd $DATA_DIR

echo "=== Downloading SIFT50M Dataset (from BIGANN/ANN_SIFT1B) ==="

# 1. 基础向量：流式解压 bigann_base.bvecs.gz，仅提取前 50M 向量
# bvecs 格式: 每向量 4+128=132 字节, 50M 向量 = 6.6 GB
if [ ! -f "sift50m_base.bvecs" ]; then
    echo "Downloading base vectors (streaming first 50M from BIGANN)..."
    wget -O - ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz \
        | gunzip -c \
        | head -c $((50000000 * 132)) \
        > sift50m_base.bvecs
    echo "Base vectors: $(wc -c < sift50m_base.bvecs) bytes (expect 6600000000)"
fi

# 2. 查询向量（与 10M 共用）
if [ ! -f "sift50m_query.bvecs" ]; then
    if [ -f "../SIFT10M/sift10m_query.bvecs" ]; then
        echo "Copying query vectors from SIFT10M..."
        cp ../SIFT10M/sift10m_query.bvecs sift50m_query.bvecs
    else
        echo "Downloading query vectors..."
        wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
        gunzip bigann_query.bvecs.gz
        mv bigann_query.bvecs sift50m_query.bvecs
    fi
fi

# 3. Ground truth (50M 子集)
if [ ! -f "sift50m_groundtruth.ivecs" ]; then
    echo "Downloading ground truth..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
    tar -xzf bigann_gnd.tar.gz
    mv gnd/idx_50M.ivecs sift50m_groundtruth.ivecs
    rm -rf gnd bigann_gnd.tar.gz
fi

ls -lh *.bvecs *.ivecs 2>/dev/null

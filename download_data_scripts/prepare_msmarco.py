#!/usr/bin/env python3
"""
MS-MARCO Passage 数据准备脚本

功能：
1. 用 sentence-transformers 编码 passages 和 queries 为 768 维 float32 向量
2. L2 归一化（cosine similarity → inner product）
3. 用 FAISS 暴力计算 top-100 ground truth
4. 输出 .fvecs / .ivecs 格式

用法：
    python prepare_msmarco.py [--data_dir ../data/MSMARCO] [--batch_size 512] [--top_k 100]

依赖：
    pip install sentence-transformers faiss-cpu numpy tqdm
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path


def load_collection(collection_path):
    """加载 MS-MARCO collection.tsv"""
    print(f"Loading collection from {collection_path}...")
    passages = {}
    with open(collection_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pid = int(parts[0])
                text = parts[1]
                passages[pid] = text
    print(f"  Loaded {len(passages)} passages")
    return passages


def load_queries(queries_path):
    """加载 MS-MARCO queries.dev.small.tsv"""
    print(f"Loading queries from {queries_path}...")
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = int(parts[0])
                text = parts[1]
                queries[qid] = text
    print(f"  Loaded {len(queries)} queries")
    return queries


def encode_texts(model, texts, batch_size=512, desc="Encoding"):
    """分批编码文本为向量"""
    from tqdm import tqdm

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False,
                                  normalize_embeddings=True)  # L2 归一化
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype(np.float32)


def write_fvecs(filename, vectors):
    """写入 .fvecs 格式（每个向量前写 int32 维度头）"""
    n, d = vectors.shape
    print(f"Writing {filename}: {n} vectors, dim={d}, "
          f"size={n * (d * 4 + 4) / 1024 / 1024:.1f} MB")
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(vectors[i].tobytes())


def write_ivecs(filename, indices):
    """写入 .ivecs 格式（每行前写 int32 个数头）"""
    n, k = indices.shape
    indices = indices.astype(np.int32)
    print(f"Writing {filename}: {n} queries, top-{k}, "
          f"size={n * (k * 4 + 4) / 1024 / 1024:.1f} MB")
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', k))
            f.write(indices[i].tobytes())


def compute_ground_truth(base_vectors, query_vectors, top_k=100):
    """用 FAISS 暴力内积搜索计算 ground truth"""
    import faiss

    n, d = base_vectors.shape
    nq = query_vectors.shape[0]
    print(f"Computing ground truth: {nq} queries against {n} vectors, top-{top_k}...")

    index = faiss.IndexFlatIP(d)  # 内积（归一化后 = cosine）
    index.add(base_vectors)

    distances, indices = index.search(query_vectors, top_k)
    print(f"  Done. Result shape: {indices.shape}")
    return indices


def main():
    parser = argparse.ArgumentParser(description="Prepare MS-MARCO data for private HNSW")
    parser.add_argument('--data_dir', type=str, default='../data/MSMARCO',
                        help='MS-MARCO 数据目录')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='编码批大小')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Ground truth top-K')
    parser.add_argument('--model_name', type=str,
                        default='sentence-transformers/msmarco-distilbert-base-tas-b',
                        help='Sentence-transformers 模型名')
    parser.add_argument('--max_passages', type=int, default=-1,
                        help='最大编码 passage 数（-1 表示全部）')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 检查原始数据
    collection_path = data_dir / 'collection.tsv'
    queries_path = data_dir / 'queries.dev.small.tsv'
    if not collection_path.exists():
        print(f"Error: {collection_path} not found. Run download_msmarco.sh first.")
        return
    if not queries_path.exists():
        print(f"Error: {queries_path} not found. Run download_msmarco.sh first.")
        return

    # 输出文件路径
    base_npy = data_dir / 'msmarco_base.npy'
    query_npy = data_dir / 'msmarco_query.npy'
    base_fvecs = data_dir / 'msmarco_base.fvecs'
    query_fvecs = data_dir / 'msmarco_query.fvecs'
    gt_ivecs = data_dir / 'msmarco_groundtruth.ivecs'

    # ========================================
    # 1. 编码 passages
    # ========================================
    if base_npy.exists():
        print(f"Loading cached passage embeddings from {base_npy}...")
        base_vectors = np.load(base_npy)
        print(f"  Shape: {base_vectors.shape}")
    else:
        from sentence_transformers import SentenceTransformer

        passages = load_collection(collection_path)

        # 按 pid 排序，保持一致的顺序
        sorted_pids = sorted(passages.keys())
        if args.max_passages > 0:
            sorted_pids = sorted_pids[:args.max_passages]
            print(f"  Using first {len(sorted_pids)} passages (--max_passages={args.max_passages})")
        passage_texts = [passages[pid] for pid in sorted_pids]

        # 保存 pid 映射（pid → index）
        pid_map_path = data_dir / 'pid_to_index.tsv'
        print(f"Saving pid mapping to {pid_map_path}...")
        with open(pid_map_path, 'w') as f:
            for idx, pid in enumerate(sorted_pids):
                f.write(f"{pid}\t{idx}\n")

        print(f"Loading model: {args.model_name}...")
        model = SentenceTransformer(args.model_name)

        base_vectors = encode_texts(model, passage_texts,
                                    batch_size=args.batch_size,
                                    desc="Encoding passages")
        print(f"  Base vectors shape: {base_vectors.shape}")

        print(f"Saving to {base_npy}...")
        np.save(base_npy, base_vectors)

    # ========================================
    # 2. 编码 queries
    # ========================================
    if query_npy.exists():
        print(f"Loading cached query embeddings from {query_npy}...")
        query_vectors = np.load(query_npy)
        print(f"  Shape: {query_vectors.shape}")
    else:
        from sentence_transformers import SentenceTransformer

        queries = load_queries(queries_path)
        sorted_qids = sorted(queries.keys())
        query_texts = [queries[qid] for qid in sorted_qids]

        # 保存 qid 映射
        qid_map_path = data_dir / 'qid_to_index.tsv'
        print(f"Saving qid mapping to {qid_map_path}...")
        with open(qid_map_path, 'w') as f:
            for idx, qid in enumerate(sorted_qids):
                f.write(f"{qid}\t{idx}\n")

        # 如果模型未加载（passages 用了缓存），重新加载
        try:
            model
        except NameError:
            print(f"Loading model: {args.model_name}...")
            model = SentenceTransformer(args.model_name)

        query_vectors = encode_texts(model, query_texts,
                                     batch_size=args.batch_size,
                                     desc="Encoding queries")
        print(f"  Query vectors shape: {query_vectors.shape}")

        print(f"Saving to {query_npy}...")
        np.save(query_npy, query_vectors)

    # ========================================
    # 3. 写 fvecs 格式
    # ========================================
    if not base_fvecs.exists():
        write_fvecs(str(base_fvecs), base_vectors)
    else:
        print(f"{base_fvecs} already exists, skipping.")

    if not query_fvecs.exists():
        write_fvecs(str(query_fvecs), query_vectors)
    else:
        print(f"{query_fvecs} already exists, skipping.")

    # ========================================
    # 4. 计算 ground truth
    # ========================================
    if not gt_ivecs.exists():
        gt_indices = compute_ground_truth(base_vectors, query_vectors, top_k=args.top_k)
        write_ivecs(str(gt_ivecs), gt_indices)
    else:
        print(f"{gt_ivecs} already exists, skipping.")

    # ========================================
    # 5. 数据统计
    # ========================================
    print("\n=== Data Summary ===")
    print(f"Base vectors:  {base_vectors.shape[0]} x {base_vectors.shape[1]}")
    print(f"Query vectors: {query_vectors.shape[0]} x {query_vectors.shape[1]}")
    print(f"Value range:   [{base_vectors.min():.4f}, {base_vectors.max():.4f}]")

    # 验证归一化
    norms = np.linalg.norm(base_vectors[:100], axis=1)
    print(f"L2 norms (first 100): mean={norms.mean():.4f}, std={norms.std():.6f}")

    print("\nOutput files:")
    for f in [base_fvecs, query_fvecs, gt_ivecs]:
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == '__main__':
    main()

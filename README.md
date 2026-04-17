# Amora

Communication-efficient private approximate nearest neighbors search (PANNS) via amortized PIR.

## Build

**Requirements:** C++17, OpenMP, CMake >= 3.10

```bash
mkdir -p build && cd build
cmake .. && make -j16
```

MT-METIS and hnswlib are included under `third_party/` and auto-detected.

## Usage

### Build Index

```bash
./build/build_index_quantized <base_vectors> <output.hnsw_q8> [M] [efConstruction] [scale] [threads] [nClusters] [graphPartition]
```

| Parameter | Pos | Description | Default |
|-----------|-----|-------------|---------|
| `M` | 3 | Graph degree (neighbors per vertex) | 16 |
| `efConstruction` | 4 | Construction search width | 200 |
| `scale` | 5 | Quantization scale factor | 0 (auto) |
| `threads` | 6 | Number of threads | 0 (auto) |
| `nClusters` | 7 | Number of partitions | -1 (sqrt(N/d)) |
| `graphPartition` | 8 | Use METIS graph partitioning | 1 (yes) |

Example:

```bash
./build/build_index_quantized data/SIFT1M/sift_base.fvecs \
    data/index/sift1m.hnsw_q8 16 200 0 16 -1 1
```

### Private Search

```bash
OMP_NUM_THREADS=16 ./build/private_search_v2 <index> <queries> <groundtruth> <stats_output> [nq] [k] [ef] [ablation]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nq` | Number of queries | All |
| `k` | Number of nearest neighbors | 10 |
| `ef` | Search beam width | 50 |
| `ablation` | Ablation flags (comma-separated) | None |

Ablation flags (combinable, e.g. `rounds_6,no_batch`):

| Flag | Description |
|------|-------------|
| `no_batch` | Disable batch PIR |
| `no_prune` | Disable centroid-based pruning |
| `no_topcand` | Disable TopCand exploration |
| `centroid_entry_K` | Override entry partition count (e.g. `centroid_entry_3`) |
| `rounds_N` | Override search iterations (e.g. `rounds_6`) |
| `stale_N` | Early stop after N stale rounds (`stale_0` disables) |

Example:

```bash
OMP_NUM_THREADS=16 ./build/private_search_v2 \
    data/index/sift1m.hnsw_q8 \
    data/SIFT1M/sift_query.fvecs \
    data/SIFT1M/sift_groundtruth.ivecs \
    stats/sift1m.stats 50 10 50
```

## Project Structure

```
private_hnsw/
├── include/                            # Headers
│   ├── private_hnsw_v2.h              # Private search core + type aliases
│   ├── hnsw_quantized.h               # Quantized HNSW index
│   ├── io.h                           # Data I/O (fvecs/bvecs/ivecs)
│   ├── quantizer.h                    # Scalar quantizer (header-only)
│   ├── graph_partitioner.h            # Graph partitioning
│   ├── subgroup/subgroup_manager.h    # Subgroup management
│   └── clustering/clustering.h        # Clustering strategies
├── src/                                # Implementations
│   ├── private_hnsw_v2.cpp            # Private search server/client
│   ├── hnsw_quantized.cpp             # Index build and serialization
│   ├── io.cpp                         # Vector file I/O
│   ├── graph_partitioner.cpp          # METIS graph partitioning
│   ├── subgroup/subgroup_manager.cpp  # Subgroup and BFS renumbering
│   └── clustering/clustering.cpp      # Clustering algorithms
├── simplepir/                          # PIR protocol layer
│   ├── embedding_pir.h/cpp            # VecPIR (templatized)
│   ├── neighbor_pir.h/cpp             # NbrPIR (templatized)
│   ├── simple_pir.h/cpp               # SimplePIR base
│   ├── simple_pir_precompute.h/cpp    # Precomputation pool
│   ├── matrix.h/cpp                   # Matrix operations (templatized)
│   ├── pir_types.h/cpp                # Type definitions (Elem32/Elem64)
│   ├── pir_math.h/cpp                 # Math ops (AVX2-optimized)
│   └── random.h/cpp                   # Random number generation
├── examples/                           # Executables
│   ├── build_index_quantized.cpp      # Index construction
│   └── private_search_v2.cpp          # Private search
├── third_party/                        # Dependencies
│   ├── mt-metis-0.7.2/               # Multi-threaded METIS
│   └── hnswlib/                       # HNSW library (header-only)
├── data/                               # Datasets and indices
└── stats/                              # Search statistics output
```

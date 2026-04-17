/**
 * @file graph_partitioner.h
 * @brief High-quality multilevel graph partitioning for HNSW
 *
 * Implements a multilevel graph partitioning algorithm similar to METIS:
 * 1. Coarsening: Contract graph using heavy edge matching
 * 2. Initial partitioning: Partition the coarsest graph
 * 3. Uncoarsening + Refinement: Expand and refine using FM algorithm
 *
 * Optimized for HNSW neighbor graphs to minimize cross-cluster edges
 * while maintaining balanced cluster sizes.
 */

#ifndef GRAPH_PARTITIONER_H
#define GRAPH_PARTITIONER_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace hnsw {

/**
 * @brief Graph representation for partitioning
 */
struct Graph {
    int numNodes;
    std::vector<std::vector<int>> adj;      // Adjacency list
    std::vector<std::vector<int>> weights;  // Edge weights (parallel to adj)
    std::vector<int> nodeWeights;           // Node weights (for coarsening)

    Graph() : numNodes(0) {}

    Graph(int n) : numNodes(n), adj(n), weights(n), nodeWeights(n, 1) {}

    void addEdge(int u, int v, int w = 1) {
        adj[u].push_back(v);
        weights[u].push_back(w);
        adj[v].push_back(u);
        weights[v].push_back(w);
    }

    int degree(int u) const { return adj[u].size(); }

    int totalWeight() const {
        return std::accumulate(nodeWeights.begin(), nodeWeights.end(), 0);
    }

    int edgeCut(const std::vector<int>& partition) const {
        int cut = 0;
        for (int u = 0; u < numNodes; u++) {
            for (size_t i = 0; i < adj[u].size(); i++) {
                int v = adj[u][i];
                if (partition[u] != partition[v]) {
                    cut += weights[u][i];
                }
            }
        }
        return cut / 2;  // Each edge counted twice
    }
};

/**
 * @brief Configuration for graph partitioning
 */
struct PartitionConfig {
    int numParts = 10;              // Number of partitions
    double imbalance = 0.03;        // Allowed imbalance (3%)
    int coarsenTo = 100;            // Coarsen until this many nodes
    int numRefinementPasses = 10;   // FM refinement passes per level
    int seed = 42;                  // Random seed
    bool verbose = false;           // Print debug info

    int maxPartSize(int totalWeight) const {
        return static_cast<int>((1.0 + imbalance) * totalWeight / numParts);
    }

    int minPartSize(int totalWeight) const {
        return static_cast<int>((1.0 - imbalance) * totalWeight / numParts);
    }
};

/**
 * @brief Result of graph partitioning
 */
struct PartitionResult {
    std::vector<int> partition;     // Node -> partition ID
    int edgeCut;                    // Number of edges cut
    std::vector<int> partSizes;     // Size of each partition
    double imbalance;               // Actual imbalance

    void print() const {
        std::cout << "=== Partition Result ===" << std::endl;
        std::cout << "Edge cut: " << edgeCut << std::endl;
        std::cout << "Imbalance: " << (imbalance * 100) << "%" << std::endl;
        std::cout << "Part sizes: [";
        for (size_t i = 0; i < partSizes.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << partSizes[i];
        }
        std::cout << "]" << std::endl;
    }
};

/**
 * @brief Multilevel graph partitioner
 */
class MultilevelPartitioner {
public:
    MultilevelPartitioner(const PartitionConfig& config = PartitionConfig())
        : config_(config), rng_(config.seed) {}

    /**
     * @brief Partition a graph into k balanced parts
     */
    PartitionResult partition(const Graph& graph);

    /**
     * @brief Build graph from HNSW neighbor structure
     */
    static Graph buildFromHNSW(
        int numNodes,
        const std::vector<std::vector<int>>& neighbors
    );

private:
    PartitionConfig config_;
    std::mt19937 rng_;

    // ========== Coarsening Phase ==========

    /**
     * @brief Coarsen graph using heavy edge matching
     */
    struct CoarsenLevel {
        Graph coarseGraph;
        std::vector<int> mapping;  // Fine node -> coarse node
        std::vector<std::vector<int>> inverseMapping;  // Coarse node -> fine nodes
    };

    CoarsenLevel coarsen(const Graph& graph);

    /**
     * @brief Heavy edge matching
     * Match nodes with their heaviest unmatched neighbor
     */
    std::vector<int> heavyEdgeMatching(const Graph& graph);

    // ========== Initial Partitioning Phase ==========

    /**
     * @brief Initial partition of coarsest graph
     * Uses greedy graph growing (GGG) heuristic
     */
    std::vector<int> initialPartition(const Graph& graph);

    /**
     * @brief Grow a partition from a seed node using BFS
     */
    void growPartition(
        const Graph& graph,
        std::vector<int>& partition,
        int partId,
        int seed,
        int targetSize,
        std::vector<bool>& assigned
    );

    // ========== Refinement Phase ==========

    /**
     * @brief Project partition to finer graph
     */
    std::vector<int> projectPartition(
        const std::vector<int>& coarsePartition,
        const CoarsenLevel& level
    );

    /**
     * @brief Fiduccia-Mattheyses (FM) refinement
     * Local search to reduce edge cut while maintaining balance
     * WARNING: O(B^2) complexity, slow for large graphs
     */
    void fmRefinement(const Graph& graph, std::vector<int>& partition);

    /**
     * @brief Label Propagation refinement (FAST)
     * O(E * iterations) complexity, much faster than FM
     * Each node adopts the label most common among neighbors
     */
    void labelPropagationRefinement(const Graph& graph, std::vector<int>& partition);

    /**
     * @brief Compute gain of moving node to another partition
     */
    int computeGain(
        const Graph& graph,
        const std::vector<int>& partition,
        int node,
        int toPart
    );

    /**
     * @brief Find best move for FM algorithm
     */
    struct Move {
        int node;
        int fromPart;
        int toPart;
        int gain;
    };

    Move findBestMove(
        const Graph& graph,
        const std::vector<int>& partition,
        const std::vector<int>& partSizes,
        const std::vector<bool>& locked,
        int maxPartSize,
        int minPartSize
    );

    // ========== Utilities ==========

    std::vector<int> computePartSizes(
        const std::vector<int>& partition,
        const std::vector<int>& nodeWeights,
        int numParts
    );

    double computeImbalance(const std::vector<int>& partSizes, int totalWeight);
};

/**
 * @brief HNSW-aware graph partitioner
 *
 * Builds graph from HNSW structure and partitions it
 */
class HNSWGraphPartitioner {
public:
    HNSWGraphPartitioner(const PartitionConfig& config = PartitionConfig())
        : config_(config) {}

    /**
     * @brief Partition HNSW nodes into balanced clusters
     * @param numNodes Total number of nodes
     * @param neighbors Level-0 neighbors for each node
     * @return Cluster assignment for each node
     */
    PartitionResult partition(
        int numNodes,
        const std::vector<std::vector<int>>& neighbors
    );

    /**
     * @brief Compute optimal number of clusters
     */
    static int computeOptimalClusters(
        int numNodes,
        int embeddingDim,
        int avgNeighborClusters,
        double imbalanceFactor = 1.5
    );

    /**
     * @brief Analyze partition quality
     */
    struct PartitionAnalysis {
        double avgNeighborClusters;  // α: average neighbor clusters per node
        double maxNeighborClusters;
        int maxClusterSize;          // K
        int minClusterSize;
        double imbalance;
        int edgeCut;
        int totalEdges;
        double edgeCutRatio;         // edgeCut / totalEdges
    };

    static PartitionAnalysis analyze(
        const std::vector<int>& partition,
        const std::vector<std::vector<int>>& neighbors,
        int numClusters
    );

private:
    PartitionConfig config_;
};

} // namespace hnsw

#endif // GRAPH_PARTITIONER_H

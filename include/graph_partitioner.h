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

struct Graph {
    int numNodes;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<int>> weights;
    std::vector<int> nodeWeights;

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
        return cut / 2;
    }
};

struct PartitionConfig {
    int numParts = 10;
    double imbalance = 0.03;
    int coarsenTo = 100;
    int numRefinementPasses = 10;
    int seed = 42;
    bool verbose = false;

    int maxPartSize(int totalWeight) const {
        return static_cast<int>((1.0 + imbalance) * totalWeight / numParts);
    }

    int minPartSize(int totalWeight) const {
        return static_cast<int>((1.0 - imbalance) * totalWeight / numParts);
    }
};

struct PartitionResult {
    std::vector<int> partition;
    int edgeCut;
    std::vector<int> partSizes;
    double imbalance;

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

class MultilevelPartitioner {
public:
    MultilevelPartitioner(const PartitionConfig& config = PartitionConfig())
        : config_(config), rng_(config.seed) {}

    PartitionResult partition(const Graph& graph);

    static Graph buildFromHNSW(
        int numNodes,
        const std::vector<std::vector<int>>& neighbors
    );

private:
    PartitionConfig config_;
    std::mt19937 rng_;

    struct CoarsenLevel {
        Graph coarseGraph;
        std::vector<int> mapping;
        std::vector<std::vector<int>> inverseMapping;
    };

    CoarsenLevel coarsen(const Graph& graph);

    std::vector<int> heavyEdgeMatching(const Graph& graph);

    std::vector<int> initialPartition(const Graph& graph);

    void growPartition(
        const Graph& graph,
        std::vector<int>& partition,
        int partId,
        int seed,
        int targetSize,
        std::vector<bool>& assigned
    );

    std::vector<int> projectPartition(
        const std::vector<int>& coarsePartition,
        const CoarsenLevel& level
    );

    void fmRefinement(const Graph& graph, std::vector<int>& partition);

    void labelPropagationRefinement(const Graph& graph, std::vector<int>& partition);

    int computeGain(
        const Graph& graph,
        const std::vector<int>& partition,
        int node,
        int toPart
    );

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

    std::vector<int> computePartSizes(
        const std::vector<int>& partition,
        const std::vector<int>& nodeWeights,
        int numParts
    );

    double computeImbalance(const std::vector<int>& partSizes, int totalWeight);
};

class HNSWGraphPartitioner {
public:
    HNSWGraphPartitioner(const PartitionConfig& config = PartitionConfig())
        : config_(config) {}

    PartitionResult partition(
        int numNodes,
        const std::vector<std::vector<int>>& neighbors
    );

    static int computeOptimalClusters(
        int numNodes,
        int embeddingDim,
        int avgNeighborClusters,
        double imbalanceFactor = 1.5
    );

    struct PartitionAnalysis {
        double avgNeighborClusters;
        double maxNeighborClusters;
        int maxClusterSize;
        int minClusterSize;
        double imbalance;
        int edgeCut;
        int totalEdges;
        double edgeCutRatio;
    };

    static PartitionAnalysis analyze(
        const std::vector<int>& partition,
        const std::vector<std::vector<int>>& neighbors,
        int numClusters
    );

private:
    PartitionConfig config_;
};

}

#endif

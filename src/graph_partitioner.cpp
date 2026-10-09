#include "../include/graph_partitioner.h"
#include <limits>
#include <cassert>
#include <map>
#include <numeric>
#include <algorithm>
#include <random>
#include <atomic>
#include <mutex>
#include <omp.h>

namespace hnsw {

PartitionResult MultilevelPartitioner::partition(const Graph& graph) {
    if (graph.numNodes == 0) {
        return PartitionResult{};
    }

    if (config_.verbose) {
        std::cout << "[Partitioner] Starting multilevel partitioning" << std::endl;
        std::cout << "  Nodes: " << graph.numNodes << std::endl;
        std::cout << "  Target parts: " << config_.numParts << std::endl;
    }

    std::vector<CoarsenLevel> levels;
    Graph currentGraph = graph;

    int minCoarseNodes = std::max(config_.coarsenTo, config_.numParts * 10);
    while (currentGraph.numNodes > minCoarseNodes) {
        auto level = coarsen(currentGraph);
        if (level.coarseGraph.numNodes >= currentGraph.numNodes * 0.9) {
            break;
        }
        if (config_.verbose) {
            std::cout << "  Coarsened: " << currentGraph.numNodes
                      << " -> " << level.coarseGraph.numNodes << std::endl;
        }
        levels.push_back(std::move(level));
        currentGraph = levels.back().coarseGraph;
    }

    if (config_.verbose) {
        std::cout << "  Coarsening levels: " << levels.size() << std::endl;
        std::cout << "  Coarsest graph: " << currentGraph.numNodes << " nodes" << std::endl;
    }

    std::vector<int> partition = initialPartition(currentGraph);

    for (int i = levels.size() - 1; i >= 0; i--) {
        partition = projectPartition(partition, levels[i]);

        const Graph& fineGraph = (i == 0) ? graph :
            (i > 0 ? levels[i - 1].coarseGraph : graph);

        labelPropagationRefinement(fineGraph, partition);

        if (config_.verbose) {
            int cut = fineGraph.edgeCut(partition);
            std::cout << "  Refined level " << i << ": edge cut = " << cut << std::endl;
        }
    }

    labelPropagationRefinement(graph, partition);

    PartitionResult result;
    result.partition = partition;
    result.edgeCut = graph.edgeCut(partition);
    result.partSizes = computePartSizes(partition, graph.nodeWeights, config_.numParts);
    result.imbalance = computeImbalance(result.partSizes, graph.totalWeight());

    if (config_.verbose) {
        std::cout << "[Partitioner] Completed" << std::endl;
        result.print();
    }

    return result;
}

Graph MultilevelPartitioner::buildFromHNSW(
    int numNodes,
    const std::vector<std::vector<int>>& neighbors
) {
    Graph graph(numNodes);

    int numThreads = omp_get_max_threads();

    std::vector<std::set<std::pair<int, int>>> threadEdgeSets(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localEdges = threadEdgeSets[tid];

        #pragma omp for schedule(dynamic, 1000)
        for (int u = 0; u < numNodes; u++) {
            for (int v : neighbors[u]) {
                if (v >= 0 && v < numNodes && u != v) {
                    auto edge = std::make_pair(std::min(u, v), std::max(u, v));
                    localEdges.insert(edge);
                }
            }
        }
    }

    std::set<std::pair<int, int>> allEdges;
    for (int t = 0; t < numThreads; t++) {
        for (const auto& edge : threadEdgeSets[t]) {
            allEdges.insert(edge);
        }
    }

    for (const auto& edge : allEdges) {
        graph.addEdge(edge.first, edge.second, 1);
    }

    return graph;
}

MultilevelPartitioner::CoarsenLevel MultilevelPartitioner::coarsen(const Graph& graph) {
    CoarsenLevel level;

    std::vector<int> match = heavyEdgeMatching(graph);

    level.mapping.resize(graph.numNodes, -1);
    int coarseId = 0;

    for (int u = 0; u < graph.numNodes; u++) {
        if (level.mapping[u] == -1) {
            int v = match[u];
            level.mapping[u] = coarseId;
            if (v != -1 && v != u) {
                level.mapping[v] = coarseId;
            }
            coarseId++;
        }
    }

    int coarseNodes = coarseId;
    level.coarseGraph = Graph(coarseNodes);
    level.inverseMapping.resize(coarseNodes);

    std::vector<std::mutex> nodeMutexes(coarseNodes);

    #pragma omp parallel for schedule(dynamic, 1000)
    for (int u = 0; u < graph.numNodes; u++) {
        int cu = level.mapping[u];
        std::lock_guard<std::mutex> lock(nodeMutexes[cu]);
        level.coarseGraph.nodeWeights[cu] += graph.nodeWeights[u];
        level.inverseMapping[cu].push_back(u);
    }

    int numThreads = omp_get_max_threads();
    std::vector<std::map<std::pair<int, int>, int>> threadEdgeMaps(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localMap = threadEdgeMaps[tid];

        #pragma omp for schedule(dynamic, 1000)
        for (int u = 0; u < graph.numNodes; u++) {
            int cu = level.mapping[u];
            for (size_t i = 0; i < graph.adj[u].size(); i++) {
                int v = graph.adj[u][i];
                int cv = level.mapping[v];
                if (cu != cv) {
                    auto edge = std::make_pair(std::min(cu, cv), std::max(cu, cv));
                    localMap[edge] += graph.weights[u][i];
                }
            }
        }
    }

    std::map<std::pair<int, int>, int> edgeWeights;
    for (int t = 0; t < numThreads; t++) {
        for (const auto& [edge, weight] : threadEdgeMaps[t]) {
            edgeWeights[edge] += weight;
        }
    }

    for (const auto& [edge, weight] : edgeWeights) {
        level.coarseGraph.adj[edge.first].push_back(edge.second);
        level.coarseGraph.weights[edge.first].push_back(weight);
        level.coarseGraph.adj[edge.second].push_back(edge.first);
        level.coarseGraph.weights[edge.second].push_back(weight);
    }

    return level;
}

std::vector<int> MultilevelPartitioner::heavyEdgeMatching(const Graph& graph) {

    std::vector<int> match(graph.numNodes, -1);
    std::vector<std::atomic<int>> proposals(graph.numNodes);
    std::vector<std::atomic<bool>> matched(graph.numNodes);

    #pragma omp parallel for
    for (int i = 0; i < graph.numNodes; i++) {
        proposals[i].store(-1, std::memory_order_relaxed);
        matched[i].store(false, std::memory_order_relaxed);
    }

    int numThreads = omp_get_max_threads();
    std::vector<std::mt19937> threadRngs(numThreads);
    for (int t = 0; t < numThreads; t++) {
        threadRngs[t].seed(config_.seed + t);
    }

    int maxRounds = 10;
    for (int round = 0; round < maxRounds; round++) {
        int unmatchedCount = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& rng = threadRngs[tid];

            #pragma omp for schedule(dynamic, 1000) reduction(+:unmatchedCount)
            for (int u = 0; u < graph.numNodes; u++) {
                if (matched[u].load(std::memory_order_relaxed)) continue;

                int bestV = -1;
                int bestWeight = -1;
                int numBest = 0;

                for (size_t i = 0; i < graph.adj[u].size(); i++) {
                    int v = graph.adj[u][i];
                    if (!matched[v].load(std::memory_order_relaxed)) {
                        int w = graph.weights[u][i];
                        if (w > bestWeight) {
                            bestWeight = w;
                            bestV = v;
                            numBest = 1;
                        } else if (w == bestWeight) {
                            numBest++;
                            std::uniform_int_distribution<int> dist(0, numBest - 1);
                            if (dist(rng) == 0) {
                                bestV = v;
                            }
                        }
                    }
                }

                proposals[u].store(bestV, std::memory_order_relaxed);
                unmatchedCount++;
            }
        }

        if (unmatchedCount == 0) break;

        #pragma omp parallel for schedule(dynamic, 1000)
        for (int u = 0; u < graph.numNodes; u++) {
            if (matched[u].load(std::memory_order_relaxed)) continue;

            int v = proposals[u].load(std::memory_order_relaxed);
            if (v == -1) {
                bool expected = false;
                if (matched[u].compare_exchange_strong(expected, true)) {
                    match[u] = u;
                }
                continue;
            }

            if (proposals[v].load(std::memory_order_relaxed) == u) {
                if (u < v) {
                    bool expected_u = false;
                    bool expected_v = false;
                    if (matched[u].compare_exchange_strong(expected_u, true)) {
                        if (matched[v].compare_exchange_strong(expected_v, true)) {
                            match[u] = v;
                            match[v] = u;
                        } else {
                            matched[u].store(false, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < graph.numNodes; i++) {
            proposals[i].store(-1, std::memory_order_relaxed);
        }
    }

    std::vector<int> remainingNodes;
    for (int u = 0; u < graph.numNodes; u++) {
        if (!matched[u].load(std::memory_order_relaxed)) {
            remainingNodes.push_back(u);
        }
    }

    std::shuffle(remainingNodes.begin(), remainingNodes.end(), threadRngs[0]);

    for (int u : remainingNodes) {
        if (matched[u].load(std::memory_order_relaxed)) continue;

        int bestV = -1;
        int bestWeight = -1;

        for (size_t i = 0; i < graph.adj[u].size(); i++) {
            int v = graph.adj[u][i];
            if (!matched[v].load(std::memory_order_relaxed) && graph.weights[u][i] > bestWeight) {
                bestWeight = graph.weights[u][i];
                bestV = v;
            }
        }

        if (bestV != -1) {
            match[u] = bestV;
            match[bestV] = u;
            matched[u].store(true, std::memory_order_relaxed);
            matched[bestV].store(true, std::memory_order_relaxed);
        } else {
            match[u] = u;
            matched[u].store(true, std::memory_order_relaxed);
        }
    }

    return match;
}

std::vector<int> MultilevelPartitioner::initialPartition(const Graph& graph) {
    std::vector<int> partition(graph.numNodes, -1);

    int totalWeight = graph.totalWeight();
    int targetWeight = (totalWeight + config_.numParts - 1) / config_.numParts;
    int maxWeight = static_cast<int>(targetWeight * (1.0 + config_.imbalance));

    std::vector<int> partWeights(config_.numParts, 0);

    std::vector<int> seeds;
    std::vector<int> distances(graph.numNodes, -1);
    std::vector<bool> isSeed(graph.numNodes, false);

    std::uniform_int_distribution<int> dist(0, graph.numNodes - 1);
    int firstSeed = dist(rng_);
    seeds.push_back(firstSeed);
    isSeed[firstSeed] = true;

    for (int p = 1; p < config_.numParts; p++) {
        std::fill(distances.begin(), distances.end(), -1);
        std::queue<int> q;
        for (int s : seeds) {
            distances[s] = 0;
            q.push(s);
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : graph.adj[u]) {
                if (distances[v] == -1) {
                    distances[v] = distances[u] + 1;
                    q.push(v);
                }
            }
        }

        int furthest = -1;
        int maxDist = -1;
        for (int u = 0; u < graph.numNodes; u++) {
            if (!isSeed[u] && distances[u] > maxDist) {
                maxDist = distances[u];
                furthest = u;
            }
        }

        if (furthest >= 0) {
            seeds.push_back(furthest);
            isSeed[furthest] = true;
        }
    }

    for (int p = 0; p < (int)seeds.size(); p++) {
        partition[seeds[p]] = p;
        partWeights[p] = graph.nodeWeights[seeds[p]];
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (int p = 0; p < config_.numParts; p++) {
            if (partWeights[p] >= maxWeight) continue;

            int bestNode = -1;
            int bestGain = -1;

            for (int u = 0; u < graph.numNodes; u++) {
                if (partition[u] != -1) continue;

                int gain = 0;
                bool hasNeighbor = false;
                for (int v : graph.adj[u]) {
                    if (partition[v] == p) {
                        gain++;
                        hasNeighbor = true;
                    }
                }

                if (hasNeighbor && gain > bestGain) {
                    bestGain = gain;
                    bestNode = u;
                }
            }

            if (bestNode >= 0 && partWeights[p] + graph.nodeWeights[bestNode] <= maxWeight) {
                partition[bestNode] = p;
                partWeights[p] += graph.nodeWeights[bestNode];
                changed = true;
            }
        }
    }

    for (int u = 0; u < graph.numNodes; u++) {
        if (partition[u] != -1) continue;

        int bestPart = -1;
        int minSize = std::numeric_limits<int>::max();

        for (int v : graph.adj[u]) {
            if (partition[v] >= 0 && partWeights[partition[v]] < minSize) {
                minSize = partWeights[partition[v]];
                bestPart = partition[v];
            }
        }

        if (bestPart == -1) {
            bestPart = 0;
            for (int p = 1; p < config_.numParts; p++) {
                if (partWeights[p] < partWeights[bestPart]) {
                    bestPart = p;
                }
            }
        }

        partition[u] = bestPart;
        partWeights[bestPart] += graph.nodeWeights[u];
    }

    return partition;
}

void MultilevelPartitioner::growPartition(
    const Graph& graph,
    std::vector<int>& partition,
    int partId,
    int seed,
    int targetSize,
    std::vector<bool>& assigned
) {
    if (assigned[seed]) return;

    int targetNodes = std::max(1, graph.numNodes / config_.numParts);

    std::priority_queue<std::pair<int, int>> pq;
    pq.push({0, seed});

    int currentSize = 0;
    int currentNodes = 0;

    while (!pq.empty() && currentSize < targetSize && currentNodes < targetNodes * 2) {
        auto [gain, u] = pq.top();
        pq.pop();

        if (assigned[u]) continue;

        partition[u] = partId;
        assigned[u] = true;
        currentSize += graph.nodeWeights[u];
        currentNodes++;

        for (int v : graph.adj[u]) {
            if (!assigned[v]) {
                int neighborGain = 0;
                for (int w : graph.adj[v]) {
                    if (partition[w] == partId) neighborGain++;
                }
                pq.push({neighborGain, v});
            }
        }
    }
}

std::vector<int> MultilevelPartitioner::projectPartition(
    const std::vector<int>& coarsePartition,
    const CoarsenLevel& level
) {
    int numFineNodes = level.mapping.size();
    std::vector<int> finePartition(numFineNodes);

    #pragma omp parallel for schedule(static)
    for (int u = 0; u < numFineNodes; u++) {
        finePartition[u] = coarsePartition[level.mapping[u]];
    }

    return finePartition;
}

void MultilevelPartitioner::fmRefinement(
    const Graph& graph,
    std::vector<int>& partition
) {
    int totalWeight = graph.totalWeight();
    int maxPartSize = config_.maxPartSize(totalWeight);
    int minPartSize = config_.minPartSize(totalWeight);

    std::vector<int> partSizes = computePartSizes(
        partition, graph.nodeWeights, config_.numParts);

    std::vector<bool> isBoundary(graph.numNodes, false);
    std::vector<int> boundaryNodes;

    auto updateBoundary = [&]() {
        boundaryNodes.clear();
        #pragma omp parallel for
        for (int u = 0; u < graph.numNodes; u++) {
            bool onBoundary = false;
            for (int v : graph.adj[u]) {
                if (partition[v] != partition[u]) {
                    onBoundary = true;
                    break;
                }
            }
            isBoundary[u] = onBoundary;
        }
        for (int u = 0; u < graph.numNodes; u++) {
            if (isBoundary[u]) boundaryNodes.push_back(u);
        }
    };

    updateBoundary();

    for (int balancePass = 0; balancePass < 5; balancePass++) {
        bool needsBalance = false;
        for (int p = 0; p < config_.numParts; p++) {
            if (partSizes[p] > maxPartSize) needsBalance = true;
        }
        if (!needsBalance) break;

        std::vector<std::vector<int>> overloadedNodes(config_.numParts);
        for (int u : boundaryNodes) {
            int p = partition[u];
            if (partSizes[p] > maxPartSize) {
                overloadedNodes[p].push_back(u);
            }
        }

        for (int p = 0; p < config_.numParts; p++) {
            for (int u : overloadedNodes[p]) {
                if (partSizes[p] <= maxPartSize) break;

                int bestTarget = -1;
                int minSize = std::numeric_limits<int>::max();

                for (int v : graph.adj[u]) {
                    int targetP = partition[v];
                    if (targetP != p &&
                        partSizes[targetP] + graph.nodeWeights[u] <= maxPartSize &&
                        partSizes[targetP] < minSize) {
                        minSize = partSizes[targetP];
                        bestTarget = targetP;
                    }
                }

                if (bestTarget >= 0) {
                    partSizes[p] -= graph.nodeWeights[u];
                    partSizes[bestTarget] += graph.nodeWeights[u];
                    partition[u] = bestTarget;
                }
            }
        }

        updateBoundary();
    }

    for (int p = 0; p < config_.numParts; p++) {
        while (partSizes[p] > maxPartSize) {
            int bestNode = -1;
            int bestTarget = -1;
            int minSize = std::numeric_limits<int>::max();

            for (int q = 0; q < config_.numParts; q++) {
                if (q != p && partSizes[q] < minSize) {
                    minSize = partSizes[q];
                    bestTarget = q;
                }
            }

            for (int u = 0; u < graph.numNodes; u++) {
                if (partition[u] == p) {
                    bestNode = u;
                    break;
                }
            }

            if (bestNode >= 0 && bestTarget >= 0) {
                partSizes[p] -= graph.nodeWeights[bestNode];
                partSizes[bestTarget] += graph.nodeWeights[bestNode];
                partition[bestNode] = bestTarget;
            } else {
                break;
            }
        }
    }

    updateBoundary();

    for (int pass = 0; pass < config_.numRefinementPasses; pass++) {
        std::vector<bool> locked(graph.numNodes, false);
        partSizes = computePartSizes(partition, graph.nodeWeights, config_.numParts);

        int moves = 0;
        int maxMoves = std::min((int)boundaryNodes.size(), graph.numNodes / 10);

        for (int iter = 0; iter < maxMoves; iter++) {
            int bestNode = -1;
            int bestToPart = -1;
            int bestGain = std::numeric_limits<int>::min();

            #pragma omp parallel
            {
                int localBestNode = -1;
                int localBestToPart = -1;
                int localBestGain = std::numeric_limits<int>::min();

                #pragma omp for nowait
                for (size_t i = 0; i < boundaryNodes.size(); i++) {
                    int u = boundaryNodes[i];
                    if (locked[u]) continue;

                    int fromPart = partition[u];
                    if (partSizes[fromPart] - graph.nodeWeights[u] < minPartSize) continue;

                    std::unordered_set<int> neighborParts;
                    for (int v : graph.adj[u]) {
                        if (partition[v] != fromPart) {
                            neighborParts.insert(partition[v]);
                        }
                    }

                    for (int toPart : neighborParts) {
                        if (partSizes[toPart] + graph.nodeWeights[u] > maxPartSize) continue;

                        int gain = computeGain(graph, partition, u, toPart);
                        if (gain > localBestGain) {
                            localBestGain = gain;
                            localBestNode = u;
                            localBestToPart = toPart;
                        }
                    }
                }

                #pragma omp critical
                {
                    if (localBestGain > bestGain) {
                        bestGain = localBestGain;
                        bestNode = localBestNode;
                        bestToPart = localBestToPart;
                    }
                }
            }

            if (bestNode == -1 || bestGain < 0) break;

            int fromPart = partition[bestNode];
            partition[bestNode] = bestToPart;
            locked[bestNode] = true;
            partSizes[fromPart] -= graph.nodeWeights[bestNode];
            partSizes[bestToPart] += graph.nodeWeights[bestNode];
            moves++;
        }

        if (moves == 0) break;
        updateBoundary();
    }
}

void MultilevelPartitioner::labelPropagationRefinement(
    const Graph& graph,
    std::vector<int>& partition
) {

    int totalWeight = graph.totalWeight();
    int maxPartSize = config_.maxPartSize(totalWeight);
    int minPartSize = config_.minPartSize(totalWeight);
    int avgPartSize = totalWeight / config_.numParts;

    std::vector<int> partSizes = computePartSizes(
        partition, graph.nodeWeights, config_.numParts);

    std::vector<int> nodeOrder(graph.numNodes);
    std::iota(nodeOrder.begin(), nodeOrder.end(), 0);

    std::mt19937 rng(config_.seed);

    int numThreads = omp_get_max_threads();

    std::vector<int> bestParts(graph.numNodes);
    std::vector<double> bestScores(graph.numNodes);

    int maxIters = 15;

    for (int iter = 0; iter < maxIters; iter++) {
        std::shuffle(nodeOrder.begin(), nodeOrder.end(), rng);

        #pragma omp parallel
        {
            std::vector<int> neighborCounts(config_.numParts, 0);

            #pragma omp for schedule(dynamic, 1000)
            for (int i = 0; i < graph.numNodes; i++) {
                int u = nodeOrder[i];
                int currentPart = partition[u];
                int nodeWeight = graph.nodeWeights[u];

                std::fill(neighborCounts.begin(), neighborCounts.end(), 0);
                for (size_t j = 0; j < graph.adj[u].size(); j++) {
                    int v = graph.adj[u][j];
                    int w = graph.weights[u][j];
                    neighborCounts[partition[v]] += w;
                }

                int bestPart = currentPart;
                double bestScore = -1e18;

                bool currentOverloaded = partSizes[currentPart] > maxPartSize;

                for (int p = 0; p < config_.numParts; p++) {
                    if (partSizes[p] + nodeWeight > maxPartSize && !currentOverloaded) continue;
                    if (p != currentPart &&
                        partSizes[currentPart] - nodeWeight < minPartSize &&
                        !currentOverloaded) continue;

                    double connectivity = neighborCounts[p];
                    double balancePenalty = 0;

                    if (p != currentPart) {
                        if (partSizes[p] > avgPartSize) {
                            balancePenalty = (partSizes[p] - avgPartSize) * 0.1;
                        }
                        if (currentOverloaded) {
                            connectivity += (partSizes[currentPart] - maxPartSize) * 0.5;
                        }
                    }

                    double score = connectivity - balancePenalty;
                    if (score > bestScore) {
                        bestScore = score;
                        bestPart = p;
                    }
                }

                bestParts[u] = bestPart;
                bestScores[u] = bestScore;
            }
        }

        int moves = 0;
        for (int i = 0; i < graph.numNodes; i++) {
            int u = nodeOrder[i];
            int currentPart = partition[u];
            int bestPart = bestParts[u];
            int nodeWeight = graph.nodeWeights[u];

            if (bestPart != currentPart) {
                bool canMove = true;
                if (partSizes[bestPart] + nodeWeight > maxPartSize) {
                    canMove = (partSizes[currentPart] > maxPartSize);
                }
                if (partSizes[currentPart] - nodeWeight < minPartSize) {
                    canMove = canMove && (partSizes[currentPart] > maxPartSize);
                }

                if (canMove) {
                    partSizes[currentPart] -= nodeWeight;
                    partSizes[bestPart] += nodeWeight;
                    partition[u] = bestPart;
                    moves++;
                }
            }
        }

        if (moves == 0) break;
    }

    for (int balancePass = 0; balancePass < 50; balancePass++) {
        int maxPart = 0;
        for (int p = 1; p < config_.numParts; p++) {
            if (partSizes[p] > partSizes[maxPart]) maxPart = p;
        }

        if (partSizes[maxPart] <= maxPartSize) break;

        bool moved = false;
        for (int u = 0; u < graph.numNodes && !moved; u++) {
            if (partition[u] != maxPart) continue;

            int nodeWeight = graph.nodeWeights[u];

            int minPart = -1;
            int minSize = std::numeric_limits<int>::max();

            for (int v : graph.adj[u]) {
                int p = partition[v];
                if (p != maxPart && partSizes[p] + nodeWeight <= maxPartSize) {
                    if (partSizes[p] < minSize) {
                        minSize = partSizes[p];
                        minPart = p;
                    }
                }
            }

            if (minPart < 0) {
                for (int p = 0; p < config_.numParts; p++) {
                    if (p != maxPart && partSizes[p] + nodeWeight <= maxPartSize) {
                        if (partSizes[p] < minSize) {
                            minSize = partSizes[p];
                            minPart = p;
                        }
                    }
                }
            }

            if (minPart >= 0) {
                partSizes[maxPart] -= nodeWeight;
                partSizes[minPart] += nodeWeight;
                partition[u] = minPart;
                moved = true;
            }
        }

        if (!moved) break;
    }
}

int MultilevelPartitioner::computeGain(
    const Graph& graph,
    const std::vector<int>& partition,
    int node,
    int toPart
) {
    int fromPart = partition[node];
    if (fromPart == toPart) return 0;

    int external = 0;
    int internal = 0;

    for (size_t i = 0; i < graph.adj[node].size(); i++) {
        int v = graph.adj[node][i];
        int w = graph.weights[node][i];
        if (partition[v] == toPart) {
            external += w;
        } else if (partition[v] == fromPart) {
            internal += w;
        }
    }

    return external - internal;
}

MultilevelPartitioner::Move MultilevelPartitioner::findBestMove(
    const Graph& graph,
    const std::vector<int>& partition,
    const std::vector<int>& partSizes,
    const std::vector<bool>& locked,
    int maxPartSize,
    int minPartSize
) {
    Move best{-1, -1, -1, std::numeric_limits<int>::min()};

    for (int u = 0; u < graph.numNodes; u++) {
        if (locked[u]) continue;

        int fromPart = partition[u];
        int nodeWeight = graph.nodeWeights[u];

        if (partSizes[fromPart] - nodeWeight < minPartSize) continue;

        std::unordered_set<int> neighborParts;
        for (int v : graph.adj[u]) {
            if (partition[v] != fromPart) {
                neighborParts.insert(partition[v]);
            }
        }

        for (int toPart : neighborParts) {
            if (partSizes[toPart] + nodeWeight > maxPartSize) continue;

            int gain = computeGain(graph, partition, u, toPart);
            if (gain > best.gain) {
                best = Move{u, fromPart, toPart, gain};
            }
        }
    }

    return best;
}

std::vector<int> MultilevelPartitioner::computePartSizes(
    const std::vector<int>& partition,
    const std::vector<int>& nodeWeights,
    int numParts
) {
    int numNodes = partition.size();
    int numThreads = omp_get_max_threads();

    std::vector<std::vector<int>> threadSizes(numThreads, std::vector<int>(numParts, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localSizes = threadSizes[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < numNodes; i++) {
            int p = partition[i];
            if (p >= 0 && p < numParts) {
                localSizes[p] += nodeWeights[i];
            }
        }
    }

    std::vector<int> sizes(numParts, 0);
    for (int t = 0; t < numThreads; t++) {
        for (int p = 0; p < numParts; p++) {
            sizes[p] += threadSizes[t][p];
        }
    }
    return sizes;
}

double MultilevelPartitioner::computeImbalance(
    const std::vector<int>& partSizes,
    int totalWeight
) {
    if (partSizes.empty() || totalWeight == 0) return 0.0;

    int avgSize = totalWeight / partSizes.size();
    int maxSize = *std::max_element(partSizes.begin(), partSizes.end());

    return static_cast<double>(maxSize - avgSize) / avgSize;
}

PartitionResult HNSWGraphPartitioner::partition(
    int numNodes,
    const std::vector<std::vector<int>>& neighbors
) {
    Graph graph = MultilevelPartitioner::buildFromHNSW(numNodes, neighbors);

    MultilevelPartitioner partitioner(config_);
    return partitioner.partition(graph);
}

int HNSWGraphPartitioner::computeOptimalClusters(
    int numNodes,
    int embeddingDim,
    int avgNeighborClusters,
    double imbalanceFactor
) {
    (void)avgNeighborClusters;
    (void)imbalanceFactor;

    int optimalC = static_cast<int>(std::sqrt((double)numNodes / (embeddingDim + 1)));

    optimalC = std::max(10, std::min(optimalC, numNodes / 100));

    return optimalC;
}

HNSWGraphPartitioner::PartitionAnalysis HNSWGraphPartitioner::analyze(
    const std::vector<int>& partition,
    const std::vector<std::vector<int>>& neighbors,
    int numClusters
) {
    PartitionAnalysis result{};
    int numNodes = partition.size();

    int numThreads = omp_get_max_threads();
    std::vector<std::vector<int>> threadClusterSizes(numThreads, std::vector<int>(numClusters, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localSizes = threadClusterSizes[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < numNodes; i++) {
            int p = partition[i];
            if (p >= 0 && p < numClusters) {
                localSizes[p]++;
            }
        }
    }

    std::vector<int> clusterSizes(numClusters, 0);
    for (int t = 0; t < numThreads; t++) {
        for (int c = 0; c < numClusters; c++) {
            clusterSizes[c] += threadClusterSizes[t][c];
        }
    }

    result.maxClusterSize = *std::max_element(clusterSizes.begin(), clusterSizes.end());
    result.minClusterSize = *std::min_element(clusterSizes.begin(), clusterSizes.end());

    int avgSize = numNodes / numClusters;
    result.imbalance = static_cast<double>(result.maxClusterSize - avgSize) / avgSize;

    double totalNeighborClusters = 0;
    double maxNeighborClusters = 0;
    long totalEdges = 0;
    long edgeCut = 0;

    #pragma omp parallel reduction(+:totalNeighborClusters,totalEdges,edgeCut) reduction(max:maxNeighborClusters)
    {
        #pragma omp for schedule(dynamic, 1000)
        for (int u = 0; u < numNodes; u++) {
            std::unordered_set<int> neighborClusters;
            int localEdges = 0;
            int localCut = 0;

            for (int v : neighbors[u]) {
                if (v >= 0 && v < numNodes) {
                    neighborClusters.insert(partition[v]);
                    localEdges++;
                    if (partition[u] != partition[v]) {
                        localCut++;
                    }
                }
            }

            totalNeighborClusters += neighborClusters.size();
            maxNeighborClusters = std::max(maxNeighborClusters, static_cast<double>(neighborClusters.size()));
            totalEdges += localEdges;
            edgeCut += localCut;
        }
    }

    result.maxNeighborClusters = maxNeighborClusters;
    result.avgNeighborClusters = totalNeighborClusters / numNodes;
    result.totalEdges = totalEdges / 2;
    result.edgeCut = edgeCut / 2;
    result.edgeCutRatio = static_cast<double>(result.edgeCut) / result.totalEdges;

    return result;
}

}

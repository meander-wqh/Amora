#include "subgroup/subgroup_manager.h"
#include "hnsw_quantized.h"
#include "clustering/clustering.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <queue>
#include <set>
#include <unordered_map>

namespace hnsw {

void SubgroupManager::initFromExistingData(
    const std::vector<int>& nodeToNewId,
    const std::vector<int>& newIdToNode,
    const std::vector<int>& clusterOffset,
    const std::vector<std::vector<int>>& subgroupOffset,
    const std::vector<int>& nodeSubgroup,
    const std::vector<int>& nodeLocalIdx,
    const std::vector<NodeNeighborInfo>& neighborInfo,
    int numClusters,
    int maxSubgroupSize) {

    nodeToNewId_ = nodeToNewId;
    newIdToNode_ = newIdToNode;
    clusterOffset_ = clusterOffset;
    subgroupOffset_ = subgroupOffset;
    nodeSubgroup_ = nodeSubgroup;
    nodeLocalIdx_ = nodeLocalIdx;
    nodeNeighborInfo_ = neighborInfo;
    numClusters_ = numClusters;
    maxSubgroupSize_ = maxSubgroupSize;

    totalSubgroups_ = 0;
    for (int c = 0; c < numClusters_; ++c) {
        if (c < (int)subgroupOffset_.size() && !subgroupOffset_[c].empty()) {
            totalSubgroups_ += (int)subgroupOffset_[c].size() - 1;
        }
    }

    subgroupToColumn_.resize(numClusters_);
    int column = 0;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        subgroupToColumn_[c].resize(numSG);
        for (int g = 0; g < numSG; ++g) {
            subgroupToColumn_[c][g] = column++;
        }
    }

    int n = (int)nodeToNewId_.size();
    minSubgroupSize_ = n;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        for (int g = 0; g < numSG; ++g) {
            int sgSize = getSubgroupSize(c, g);
            if (sgSize > 0) {
                minSubgroupSize_ = std::min(minSubgroupSize_, sgSize);
            }
        }
    }

    isBuilt_ = true;
    hasSubgrouping_ = true;

    std::cout << "[SubgroupManager] Initialized from existing data." << std::endl;
    std::cout << "  Num clusters: " << numClusters_ << std::endl;
    std::cout << "  Total subgroups: " << totalSubgroups_ << std::endl;
    std::cout << "  Max subgroup size: " << maxSubgroupSize_ << std::endl;
}

void SubgroupManager::initFromExistingDataMove(
    std::vector<int>&& nodeToNewId,
    std::vector<int>&& newIdToNode,
    std::vector<int>&& clusterOffset,
    std::vector<std::vector<int>>&& subgroupOffset,
    std::vector<int>&& nodeSubgroup,
    std::vector<int>&& nodeLocalIdx,
    std::vector<NodeNeighborInfo>&& neighborInfo,
    int numClusters,
    int maxSubgroupSize) {

    nodeToNewId_ = std::move(nodeToNewId);
    newIdToNode_ = std::move(newIdToNode);
    clusterOffset_ = std::move(clusterOffset);
    subgroupOffset_ = std::move(subgroupOffset);
    nodeSubgroup_ = std::move(nodeSubgroup);
    nodeLocalIdx_ = std::move(nodeLocalIdx);
    nodeNeighborInfo_ = std::move(neighborInfo);
    numClusters_ = numClusters;
    maxSubgroupSize_ = maxSubgroupSize;

    totalSubgroups_ = 0;
    for (int c = 0; c < numClusters_; ++c) {
        if (c < (int)subgroupOffset_.size() && !subgroupOffset_[c].empty()) {
            totalSubgroups_ += (int)subgroupOffset_[c].size() - 1;
        }
    }

    subgroupToColumn_.resize(numClusters_);
    int column = 0;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        subgroupToColumn_[c].resize(numSG);
        for (int g = 0; g < numSG; ++g) {
            subgroupToColumn_[c][g] = column++;
        }
    }

    int n = (int)nodeToNewId_.size();
    minSubgroupSize_ = n;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        for (int g = 0; g < numSG; ++g) {
            int sgSize = getSubgroupSize(c, g);
            if (sgSize > 0) {
                minSubgroupSize_ = std::min(minSubgroupSize_, sgSize);
            }
        }
    }

    isBuilt_ = true;
    hasSubgrouping_ = true;

    std::cout << "[SubgroupManager] Initialized from existing data (move)." << std::endl;
    std::cout << "  Num clusters: " << numClusters_ << std::endl;
    std::cout << "  Total subgroups: " << totalSubgroups_ << std::endl;
    std::cout << "  Max subgroup size: " << maxSubgroupSize_ << std::endl;
}

void SubgroupManager::setRenumberData(
    const std::vector<int>& nodeToNewId,
    const std::vector<int>& newIdToNode,
    const std::vector<int>& clusterOffset,
    int numClusters) {
    nodeToNewId_ = nodeToNewId;
    newIdToNode_ = newIdToNode;
    clusterOffset_ = clusterOffset;
    numClusters_ = numClusters;
    isBuilt_ = true;
}

void SubgroupManager::renumberNodesByCluster(
    const HNSWQuantizedIndex& index,
    const std::vector<int>& assignments,
    int numClusters) {

    int n = index.ntotal.load();
    numClusters_ = numClusters;

    std::cout << "Renumbering " << n << " nodes by cluster order..." << std::endl;

    nodeToNewId_.resize(n);
    newIdToNode_.resize(n);
    clusterOffset_.resize(numClusters + 1);

    std::vector<int> clusterSizes(numClusters, 0);
    for (int i = 0; i < n; ++i) {
        if (assignments[i] >= 0 && assignments[i] < numClusters) {
            clusterSizes[assignments[i]]++;
        }
    }

    clusterOffset_[0] = 0;
    for (int c = 0; c < numClusters; ++c) {
        clusterOffset_[c + 1] = clusterOffset_[c] + clusterSizes[c];
    }

    std::vector<std::vector<int>> clusterNodes(numClusters);
    for (int c = 0; c < numClusters; ++c) {
        clusterNodes[c].reserve(clusterSizes[c]);
    }
    for (int i = 0; i < n; ++i) {
        if (assignments[i] >= 0 && assignments[i] < numClusters) {
            clusterNodes[assignments[i]].push_back(i);
        }
    }

    std::vector<std::vector<int>> clusterBfsOrder(numClusters);

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < numClusters; ++c) {
        const auto& nodes = clusterNodes[c];
        int cSize = (int)nodes.size();
        if (cSize == 0) continue;

        std::unordered_map<int, int> oldIdToLocal;
        for (int i = 0; i < cSize; ++i) {
            oldIdToLocal[nodes[i]] = i;
        }

        std::vector<std::vector<int>> adjList(cSize);
        std::vector<int> degree(cSize, 0);
        for (int i = 0; i < cSize; ++i) {
            int oldId = nodes[i];
            if (oldId >= 0 && oldId < (int)index.neighbors.size() && !index.neighbors[oldId].empty()) {
                for (int32_t nbrOldId : index.neighbors[oldId][0]) {
                    if (nbrOldId < 0 || nbrOldId >= n) continue;
                    auto it = oldIdToLocal.find(nbrOldId);
                    if (it != oldIdToLocal.end()) {
                        adjList[i].push_back(it->second);
                        degree[i]++;
                    }
                }
            }
        }

        int seed = 0;
        for (int i = 1; i < cSize; ++i) {
            if (degree[i] > degree[seed]) seed = i;
        }

        auto& bfsOrder = clusterBfsOrder[c];
        bfsOrder.reserve(cSize);
        std::vector<bool> visited(cSize, false);

        std::queue<int> bfsQueue;
        bfsQueue.push(seed);
        visited[seed] = true;

        while ((int)bfsOrder.size() < cSize) {
            while (!bfsQueue.empty()) {
                int cur = bfsQueue.front();
                bfsQueue.pop();
                bfsOrder.push_back(cur);

                for (int nbr : adjList[cur]) {
                    if (!visited[nbr]) {
                        visited[nbr] = true;
                        bfsQueue.push(nbr);
                    }
                }
            }

            if ((int)bfsOrder.size() < cSize) {
                int nextSeed = -1;
                for (int i = 0; i < cSize; ++i) {
                    if (!visited[i]) {
                        if (nextSeed < 0 || degree[i] > degree[nextSeed]) {
                            nextSeed = i;
                        }
                    }
                }
                if (nextSeed >= 0) {
                    visited[nextSeed] = true;
                    bfsQueue.push(nextSeed);
                }
            }
        }
    }

    int newId = 0;
    for (int c = 0; c < numClusters; ++c) {
        const auto& nodes = clusterNodes[c];
        for (int localIdx : clusterBfsOrder[c]) {
            int oldId = nodes[localIdx];
            nodeToNewId_[oldId] = newId;
            newIdToNode_[newId] = oldId;
            newId++;
        }
    }

    isBuilt_ = true;

    std::cout << "Node renumbering complete." << std::endl;
    std::cout << "  Cluster sizes: ";
    int maxSize = 0, minSize = n;
    for (int c = 0; c < numClusters; ++c) {
        int size = clusterOffset_[c + 1] - clusterOffset_[c];
        maxSize = std::max(maxSize, size);
        minSize = std::min(minSize, size);
    }
    std::cout << "min=" << minSize << ", max=" << maxSize
              << ", avg=" << std::fixed << std::setprecision(1)
              << (double)n / numClusters << std::endl;
}

void SubgroupManager::buildSubgroups(
    const HNSWQuantizedIndex& index,
    const std::vector<int>& assignments,
    int targetSubgroupSize) {

    if (!isBuilt_) {
        throw std::runtime_error("buildSubgroups requires node renumbering first");
    }

    int n = index.ntotal.load();
    int target = targetSubgroupSize;
    std::cout << "Building subgroups with sequential partitioning, target size " << target << "..." << std::endl;

    subgroupOffset_.resize(numClusters_);
    nodeSubgroup_.resize(n, -1);
    nodeLocalIdx_.resize(n, -1);

    totalSubgroups_ = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:totalSubgroups_)
    for (int c = 0; c < numClusters_; ++c) {
        int clusterStart = clusterOffset_[c];
        int clusterEnd = clusterOffset_[c + 1];
        int clusterSize = clusterEnd - clusterStart;

        if (clusterSize == 0) {
            subgroupOffset_[c].push_back(0);
            continue;
        }

        int numSG = (clusterSize + target - 1) / target;
        numSG = std::max(1, numSG);

        subgroupOffset_[c].resize(numSG + 1);
        for (int g = 0; g <= numSG; ++g) {
            subgroupOffset_[c][g] = std::min(g * target, clusterSize);
        }

        for (int i = 0; i < clusterSize; ++i) {
            int newId = clusterStart + i;
            nodeSubgroup_[newId] = i / target;
            nodeLocalIdx_[newId] = i % target;
        }

        totalSubgroups_ += numSG;
    }

    maxSubgroupSize_ = target;
    minSubgroupSize_ = n;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = (int)subgroupOffset_[c].size() - 1;
        for (int g = 0; g < numSG; ++g) {
            int sgSize = subgroupOffset_[c][g + 1] - subgroupOffset_[c][g];
            if (sgSize > 0) {
                minSubgroupSize_ = std::min(minSubgroupSize_, sgSize);
            }
        }
    }
    hasSubgrouping_ = true;

    subgroupToColumn_.resize(numClusters_);
    int column = 0;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        subgroupToColumn_[c].resize(numSG);
        for (int g = 0; g < numSG; ++g) {
            subgroupToColumn_[c][g] = column++;
        }
    }

    std::cout << "Subgroup building complete (sequential partitioning)." << std::endl;
    std::cout << "  Total subgroups: " << totalSubgroups_ << std::endl;
    std::cout << "  Avg subgroups per cluster: " << std::fixed << std::setprecision(1)
              << (double)totalSubgroups_ / numClusters_ << std::endl;
    std::cout << "  Subgroup size: min=" << minSubgroupSize_
              << ", max=" << maxSubgroupSize_ << " (= targetSubgroupSize)" << std::endl;
}

void SubgroupManager::buildNeighborInfo(
    const HNSWQuantizedIndex& index,
    const std::vector<int>& assignments) {

    if (!hasSubgrouping_) {
        throw std::runtime_error("buildNeighborInfo requires subgroup data");
    }

    int n = index.ntotal.load();
    std::cout << "Building neighbor info for " << n << " nodes..." << std::endl;

    nodeNeighborInfo_.resize(n);

    std::atomic<int64_t> crossClusterEdges{0};
    std::atomic<int64_t> totalEdges{0};
    std::atomic<int> maxNeighborGroups{0};
    std::atomic<int64_t> sumNeighborGroups{0};

    #pragma omp parallel
    {
        std::map<std::pair<int,int>, std::vector<int>> neighborsBySubgroup;

        int64_t localCrossEdges = 0;
        int64_t localTotalEdges = 0;
        int localMaxGroups = 0;
        int64_t localSumGroups = 0;

        #pragma omp for schedule(dynamic, 4096)
        for (int newId = 0; newId < n; ++newId) {
            int oldId = newIdToNode_[newId];
            int myCluster = assignments[oldId];

            neighborsBySubgroup.clear();

            if (oldId >= 0 && oldId < (int)index.neighbors.size() && !index.neighbors[oldId].empty()) {
                for (int32_t nbrOldId : index.neighbors[oldId][0]) {
                    if (nbrOldId < 0 || nbrOldId >= n) continue;

                    localTotalEdges++;
                    int nbrCluster = assignments[nbrOldId];
                    int nbrNewId = nodeToNewId_[nbrOldId];

                    if (nbrCluster != myCluster) {
                        localCrossEdges++;
                    }

                    int nbrSubgroup = getSubgroup(nbrNewId);
                    int nbrLocalIdx = getLocalIndex(nbrNewId);

                    auto key = std::make_pair(nbrCluster, nbrSubgroup);
                    neighborsBySubgroup[key].push_back(nbrLocalIdx);
                }
            }

            nodeNeighborInfo_[newId].groups.clear();
            for (auto& [key, indices] : neighborsBySubgroup) {
                NeighborGroup group;
                group.cluster = key.first;
                group.subgroup = key.second;
                group.localIndices = std::move(indices);
                nodeNeighborInfo_[newId].groups.push_back(std::move(group));
            }

            int numGroups = (int)nodeNeighborInfo_[newId].groups.size();
            localMaxGroups = std::max(localMaxGroups, numGroups);
            localSumGroups += numGroups;
        }

        totalEdges.fetch_add(localTotalEdges);
        crossClusterEdges.fetch_add(localCrossEdges);
        sumNeighborGroups.fetch_add(localSumGroups);
        int prev = maxNeighborGroups.load();
        while (localMaxGroups > prev &&
               !maxNeighborGroups.compare_exchange_weak(prev, localMaxGroups));
    }

    double avgNeighborGroups = (double)sumNeighborGroups.load() / n;

    std::cout << "Neighbor info building complete." << std::endl;
    std::cout << "  Total edges: " << totalEdges.load() << std::endl;
    std::cout << "  Cross-cluster edges: " << crossClusterEdges.load()
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * crossClusterEdges.load() / totalEdges.load()) << "%)" << std::endl;
    std::cout << "  Neighbor groups per node: avg=" << std::fixed << std::setprecision(1)
              << avgNeighborGroups << ", max=" << maxNeighborGroups.load() << std::endl;
}

int SubgroupManager::getCluster(int newId) const {
    if (!isBuilt_ || newId < 0 || newId >= (int)newIdToNode_.size()) {
        return -1;
    }

    auto it = std::upper_bound(clusterOffset_.begin(), clusterOffset_.end(), newId);
    if (it == clusterOffset_.begin()) return -1;
    return (int)(it - clusterOffset_.begin() - 1);
}

int SubgroupManager::getSubgroup(int newId) const {
    if (!hasSubgrouping_) return -1;

    int cluster = getCluster(newId);
    if (cluster < 0 || cluster >= numClusters_) return -1;

    int localInCluster = newId - clusterOffset_[cluster];
    const auto& offsets = subgroupOffset_[cluster];

    auto it = std::upper_bound(offsets.begin(), offsets.end(), localInCluster);
    if (it == offsets.begin()) return -1;
    return (int)(it - offsets.begin() - 1);
}

int SubgroupManager::getLocalIndex(int newId) const {
    if (!hasSubgrouping_) return -1;

    int cluster = getCluster(newId);
    if (cluster < 0 || cluster >= numClusters_) return -1;

    int subgroup = getSubgroup(newId);
    if (subgroup < 0 || subgroup >= (int)subgroupOffset_[cluster].size() - 1) return -1;

    int localInCluster = newId - clusterOffset_[cluster];
    return localInCluster - subgroupOffset_[cluster][subgroup];
}

std::pair<int, int> SubgroupManager::getClusterAndSubgroup(int newId) const {
    return {getCluster(newId), getSubgroup(newId)};
}

int SubgroupManager::getSubgroupSize(int cluster, int subgroup) const {
    if (!hasSubgrouping_) return 0;
    if (cluster < 0 || cluster >= numClusters_) return 0;
    if (subgroup < 0 || subgroup >= (int)subgroupOffset_[cluster].size() - 1) return 0;

    return subgroupOffset_[cluster][subgroup + 1] - subgroupOffset_[cluster][subgroup];
}

int SubgroupManager::getSubgroupColumn(int cluster, int subgroup) const {
    if (!hasSubgrouping_) return -1;
    if (cluster < 0 || cluster >= (int)subgroupToColumn_.size()) return -1;
    if (subgroup < 0 || subgroup >= (int)subgroupToColumn_[cluster].size()) return -1;

    return subgroupToColumn_[cluster][subgroup];
}

int SubgroupManager::getNumSubgroupsInCluster(int cluster) const {
    if (!hasSubgrouping_) return 0;
    if (cluster < 0 || cluster >= (int)subgroupOffset_.size()) return 0;
    return (int)subgroupOffset_[cluster].size() - 1;
}

void SubgroupManager::save(std::ofstream& ofs) const {
    ofs.write(reinterpret_cast<const char*>(&hasSubgrouping_), sizeof(hasSubgrouping_));

    if (!hasSubgrouping_) return;

    int n = (int)nodeToNewId_.size();

    ofs.write(reinterpret_cast<const char*>(&numClusters_), sizeof(numClusters_));
    ofs.write(reinterpret_cast<const char*>(&totalSubgroups_), sizeof(totalSubgroups_));
    ofs.write(reinterpret_cast<const char*>(&maxSubgroupSize_), sizeof(maxSubgroupSize_));

    ofs.write(reinterpret_cast<const char*>(nodeToNewId_.data()), n * sizeof(int));
    ofs.write(reinterpret_cast<const char*>(newIdToNode_.data()), n * sizeof(int));
    ofs.write(reinterpret_cast<const char*>(nodeSubgroup_.data()), n * sizeof(int));
    ofs.write(reinterpret_cast<const char*>(nodeLocalIdx_.data()), n * sizeof(int));

    int numOffsets = (int)clusterOffset_.size();
    ofs.write(reinterpret_cast<const char*>(&numOffsets), sizeof(numOffsets));
    ofs.write(reinterpret_cast<const char*>(clusterOffset_.data()), numOffsets * sizeof(int));

    for (int c = 0; c < numClusters_; ++c) {
        int numSubOffsets = (int)subgroupOffset_[c].size();
        ofs.write(reinterpret_cast<const char*>(&numSubOffsets), sizeof(numSubOffsets));
        ofs.write(reinterpret_cast<const char*>(subgroupOffset_[c].data()), numSubOffsets * sizeof(int));
    }

    for (int i = 0; i < n; ++i) {
        int numGroups = (int)nodeNeighborInfo_[i].groups.size();
        ofs.write(reinterpret_cast<const char*>(&numGroups), sizeof(numGroups));

        for (const auto& group : nodeNeighborInfo_[i].groups) {
            ofs.write(reinterpret_cast<const char*>(&group.cluster), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&group.subgroup), sizeof(int));
            int numIndices = (int)group.localIndices.size();
            ofs.write(reinterpret_cast<const char*>(&numIndices), sizeof(numIndices));
            if (numIndices > 0) {
                ofs.write(reinterpret_cast<const char*>(group.localIndices.data()), numIndices * sizeof(int));
            }
        }
    }

    std::cout << "Subgroup info saved." << std::endl;
}

void SubgroupManager::load(std::ifstream& ifs) {
    ifs.read(reinterpret_cast<char*>(&hasSubgrouping_), sizeof(hasSubgrouping_));

    if (!hasSubgrouping_) return;

    ifs.read(reinterpret_cast<char*>(&numClusters_), sizeof(numClusters_));
    ifs.read(reinterpret_cast<char*>(&totalSubgroups_), sizeof(totalSubgroups_));
    ifs.read(reinterpret_cast<char*>(&maxSubgroupSize_), sizeof(maxSubgroupSize_));

    int numOffsets;
    ifs.read(reinterpret_cast<char*>(&numOffsets), sizeof(numOffsets));
    clusterOffset_.resize(numOffsets);
    ifs.read(reinterpret_cast<char*>(clusterOffset_.data()), numOffsets * sizeof(int));

    int n = clusterOffset_.back();

    ifs.seekg(sizeof(hasSubgrouping_) + sizeof(numClusters_) + sizeof(totalSubgroups_) + sizeof(maxSubgroupSize_));

    nodeToNewId_.resize(n);
    newIdToNode_.resize(n);
    nodeSubgroup_.resize(n);
    nodeLocalIdx_.resize(n);

    ifs.read(reinterpret_cast<char*>(nodeToNewId_.data()), n * sizeof(int));
    ifs.read(reinterpret_cast<char*>(newIdToNode_.data()), n * sizeof(int));
    ifs.read(reinterpret_cast<char*>(nodeSubgroup_.data()), n * sizeof(int));
    ifs.read(reinterpret_cast<char*>(nodeLocalIdx_.data()), n * sizeof(int));

    ifs.read(reinterpret_cast<char*>(&numOffsets), sizeof(numOffsets));
    clusterOffset_.resize(numOffsets);
    ifs.read(reinterpret_cast<char*>(clusterOffset_.data()), numOffsets * sizeof(int));

    subgroupOffset_.resize(numClusters_);
    for (int c = 0; c < numClusters_; ++c) {
        int numSubOffsets;
        ifs.read(reinterpret_cast<char*>(&numSubOffsets), sizeof(numSubOffsets));
        subgroupOffset_[c].resize(numSubOffsets);
        ifs.read(reinterpret_cast<char*>(subgroupOffset_[c].data()), numSubOffsets * sizeof(int));
    }

    nodeNeighborInfo_.resize(n);
    for (int i = 0; i < n; ++i) {
        int numGroups;
        ifs.read(reinterpret_cast<char*>(&numGroups), sizeof(numGroups));
        nodeNeighborInfo_[i].groups.resize(numGroups);

        for (int g = 0; g < numGroups; ++g) {
            ifs.read(reinterpret_cast<char*>(&nodeNeighborInfo_[i].groups[g].cluster), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&nodeNeighborInfo_[i].groups[g].subgroup), sizeof(int));
            int numIndices;
            ifs.read(reinterpret_cast<char*>(&numIndices), sizeof(numIndices));
            nodeNeighborInfo_[i].groups[g].localIndices.resize(numIndices);
            if (numIndices > 0) {
                ifs.read(reinterpret_cast<char*>(nodeNeighborInfo_[i].groups[g].localIndices.data()), numIndices * sizeof(int));
            }
        }
    }

    subgroupToColumn_.resize(numClusters_);
    int column = 0;
    for (int c = 0; c < numClusters_; ++c) {
        int numSG = getNumSubgroupsInCluster(c);
        subgroupToColumn_[c].resize(numSG);
        for (int g = 0; g < numSG; ++g) {
            subgroupToColumn_[c][g] = column++;
        }
    }

    isBuilt_ = true;

    std::cout << "Subgroup info loaded." << std::endl;
}

void SubgroupManager::printStats() const {
    std::cout << "=== SubgroupManager Stats ===" << std::endl;
    std::cout << "  Built: " << (isBuilt_ ? "yes" : "no") << std::endl;
    std::cout << "  Has subgrouping: " << (hasSubgrouping_ ? "yes" : "no") << std::endl;

    if (hasSubgrouping_) {
        std::cout << "  Num clusters: " << numClusters_ << std::endl;
        std::cout << "  Total subgroups: " << totalSubgroups_ << std::endl;
        std::cout << "  Max subgroup size: " << maxSubgroupSize_ << std::endl;
        std::cout << "  Min subgroup size: " << minSubgroupSize_ << std::endl;
    }

    std::cout << "=============================" << std::endl;
}

}

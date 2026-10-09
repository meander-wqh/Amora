#pragma once

#include <vector>
#include <map>
#include <fstream>
#include <string>

namespace hnsw {

class HNSWQuantizedIndex;
struct ClusteringResult;

struct NeighborGroup {
    int cluster;
    int subgroup;
    std::vector<int> localIndices;
};

struct NodeNeighborInfo {
    std::vector<NeighborGroup> groups;
};

class SubgroupManager {
public:
    SubgroupManager() = default;

    void initFromExistingData(
        const std::vector<int>& nodeToNewId,
        const std::vector<int>& newIdToNode,
        const std::vector<int>& clusterOffset,
        const std::vector<std::vector<int>>& subgroupOffset,
        const std::vector<int>& nodeSubgroup,
        const std::vector<int>& nodeLocalIdx,
        const std::vector<NodeNeighborInfo>& neighborInfo,
        int numClusters,
        int maxSubgroupSize);

    void initFromExistingDataMove(
        std::vector<int>&& nodeToNewId,
        std::vector<int>&& newIdToNode,
        std::vector<int>&& clusterOffset,
        std::vector<std::vector<int>>&& subgroupOffset,
        std::vector<int>&& nodeSubgroup,
        std::vector<int>&& nodeLocalIdx,
        std::vector<NodeNeighborInfo>&& neighborInfo,
        int numClusters,
        int maxSubgroupSize);

    void renumberNodesByCluster(const HNSWQuantizedIndex& index,
                                const std::vector<int>& assignments,
                                int numClusters);

    void setRenumberData(
        const std::vector<int>& nodeToNewId,
        const std::vector<int>& newIdToNode,
        const std::vector<int>& clusterOffset,
        int numClusters);

    void buildSubgroups(const HNSWQuantizedIndex& index,
                       const std::vector<int>& assignments,
                       int targetSubgroupSize);

    void buildNeighborInfo(const HNSWQuantizedIndex& index,
                          const std::vector<int>& assignments);

    int getNewId(int originalId) const {
        if (originalId < 0 || originalId >= (int)nodeToNewId_.size()) return -1;
        return nodeToNewId_[originalId];
    }

    int getOriginalId(int newId) const {
        if (newId < 0 || newId >= (int)newIdToNode_.size()) return -1;
        return newIdToNode_[newId];
    }

    int getCluster(int newId) const;

    int getSubgroup(int newId) const;

    int getLocalIndex(int newId) const;

    std::pair<int, int> getClusterAndSubgroup(int newId) const;

    int getTotalSubgroups() const { return totalSubgroups_; }

    int getMaxSubgroupSize() const { return maxSubgroupSize_; }

    int getSubgroupSize(int cluster, int subgroup) const;

    int getSubgroupColumn(int cluster, int subgroup) const;

    int getNumSubgroupsInCluster(int cluster) const;

    int getClusterOffset(int cluster) const {
        if (cluster < 0 || cluster >= (int)clusterOffset_.size()) return -1;
        return clusterOffset_[cluster];
    }

    const NodeNeighborInfo& getNeighborInfo(int newId) const {
        static NodeNeighborInfo empty;
        if (newId < 0 || newId >= (int)nodeNeighborInfo_.size()) return empty;
        return nodeNeighborInfo_[newId];
    }

    bool isBuilt() const { return isBuilt_; }
    bool hasSubgrouping() const { return hasSubgrouping_; }
    int getNumClusters() const { return numClusters_; }

    void save(std::ofstream& ofs) const;
    void load(std::ifstream& ifs);

    void printStats() const;

    const std::vector<int>& getNodeToNewId() const { return nodeToNewId_; }
    const std::vector<int>& getNewIdToNode() const { return newIdToNode_; }
    const std::vector<int>& getClusterOffsets() const { return clusterOffset_; }
    const std::vector<std::vector<int>>& getSubgroupOffsets() const { return subgroupOffset_; }
    const std::vector<int>& getNodeSubgroup() const { return nodeSubgroup_; }
    const std::vector<int>& getNodeLocalIdx() const { return nodeLocalIdx_; }
    const std::vector<NodeNeighborInfo>& getAllNeighborInfo() const { return nodeNeighborInfo_; }

    std::vector<int> moveNodeToNewId() { return std::move(nodeToNewId_); }
    std::vector<int> moveNewIdToNode() { return std::move(newIdToNode_); }
    std::vector<int> moveClusterOffsets() { return std::move(clusterOffset_); }
    std::vector<std::vector<int>> moveSubgroupOffsets() { return std::move(subgroupOffset_); }
    std::vector<int> moveNodeSubgroup() { return std::move(nodeSubgroup_); }
    std::vector<int> moveNodeLocalIdx() { return std::move(nodeLocalIdx_); }
    std::vector<NodeNeighborInfo> moveAllNeighborInfo() { return std::move(nodeNeighborInfo_); }

private:
    bool isBuilt_ = false;
    bool hasSubgrouping_ = false;
    int numClusters_ = 0;
    int totalSubgroups_ = 0;
    int maxSubgroupSize_ = 0;
    int minSubgroupSize_ = 0;

    std::vector<int> nodeToNewId_;
    std::vector<int> newIdToNode_;

    std::vector<int> clusterOffset_;
    std::vector<std::vector<int>> subgroupOffset_;

    std::vector<int> nodeSubgroup_;
    std::vector<int> nodeLocalIdx_;

    std::vector<std::vector<int>> subgroupToColumn_;

    std::vector<NodeNeighborInfo> nodeNeighborInfo_;
};

}

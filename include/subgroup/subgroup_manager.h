#pragma once

#include <vector>
#include <map>
#include <fstream>
#include <string>

namespace hnsw {

// 前向声明
class HNSWQuantizedIndex;
struct ClusteringResult;

/**
 * @brief 邻居组信息
 *
 * 一个节点的邻居可能分布在多个(聚类,子组)对中
 */
struct NeighborGroup {
    int cluster;                        // 邻居所在聚类
    int subgroup;                       // 邻居所在子组
    std::vector<int> localIndices;      // 邻居在子组内的局部索引
};

/**
 * @brief 节点邻居信息
 */
struct NodeNeighborInfo {
    std::vector<NeighborGroup> groups;  // 邻居按(聚类,子组)分组
};

/**
 * @brief 子组管理器
 *
 * 负责管理聚类内的子组划分和节点映射
 */
class SubgroupManager {
public:
    SubgroupManager() = default;

    /**
     * @brief 从 HNSWQuantizedIndex 现有数据初始化
     *
     * 用于从已有索引的子组数据创建 SubgroupManager
     *
     * @param nodeToNewId 原始ID -> 新ID 映射
     * @param newIdToNode 新ID -> 原始ID 映射
     * @param clusterOffset 聚类偏移数组
     * @param subgroupOffset 子组偏移数组
     * @param nodeSubgroup 节点子组映射
     * @param nodeLocalIdx 节点在子组内的局部索引
     * @param neighborInfo 邻居信息
     * @param numClusters 聚类数量
     * @param maxSubgroupSize 最大子组大小
     */
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

    // Move 版本：零拷贝接管所有数据
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

    /**
     * @brief 按聚类顺序重新编号节点
     *
     * @param index HNSW 索引
     * @param assignments 聚类分配（每个节点的聚类ID）
     * @param numClusters 聚类数量
     */
    void renumberNodesByCluster(const HNSWQuantizedIndex& index,
                                const std::vector<int>& assignments,
                                int numClusters);

    /**
     * @brief 设置已有的重编号数据（跳过 BFS 重编号）
     *
     * 用于 buildSubgroups 等场景，避免重复执行 BFS 重编号
     */
    void setRenumberData(
        const std::vector<int>& nodeToNewId,
        const std::vector<int>& newIdToNode,
        const std::vector<int>& clusterOffset,
        int numClusters);

    /**
     * @brief 构建子组划分
     *
     * 使用贪心BFS算法，将聚类内节点划分为子组
     *
     * @param index HNSW 索引
     * @param assignments 聚类分配
     * @param targetSubgroupSize 目标子组大小
     */
    void buildSubgroups(const HNSWQuantizedIndex& index,
                       const std::vector<int>& assignments,
                       int targetSubgroupSize);

    /**
     * @brief 构建邻居信息
     *
     * 为每个节点生成邻居子组信息
     *
     * @param index HNSW 索引
     * @param assignments 聚类分配
     */
    void buildNeighborInfo(const HNSWQuantizedIndex& index,
                          const std::vector<int>& assignments);

    // ========== 节点映射 ==========

    /** 原始ID -> 新ID */
    int getNewId(int originalId) const {
        if (originalId < 0 || originalId >= (int)nodeToNewId_.size()) return -1;
        return nodeToNewId_[originalId];
    }

    /** 新ID -> 原始ID */
    int getOriginalId(int newId) const {
        if (newId < 0 || newId >= (int)newIdToNode_.size()) return -1;
        return newIdToNode_[newId];
    }

    // ========== 子组查询 ==========

    /** 根据新ID获取聚类 */
    int getCluster(int newId) const;

    /** 根据新ID获取子组 */
    int getSubgroup(int newId) const;

    /** 根据新ID获取在子组内的局部索引 */
    int getLocalIndex(int newId) const;

    /** 获取(聚类, 子组)对 */
    std::pair<int, int> getClusterAndSubgroup(int newId) const;

    // ========== 子组信息 ==========

    /** 获取总子组数 */
    int getTotalSubgroups() const { return totalSubgroups_; }

    /** 获取最大子组大小 */
    int getMaxSubgroupSize() const { return maxSubgroupSize_; }

    /** 获取指定子组大小 */
    int getSubgroupSize(int cluster, int subgroup) const;

    /** 获取子组的PIR列索引 */
    int getSubgroupColumn(int cluster, int subgroup) const;

    /** 获取聚类内子组数量 */
    int getNumSubgroupsInCluster(int cluster) const;

    /** 获取聚类偏移量 */
    int getClusterOffset(int cluster) const {
        if (cluster < 0 || cluster >= (int)clusterOffset_.size()) return -1;
        return clusterOffset_[cluster];
    }

    /** 获取节点邻居信息 */
    const NodeNeighborInfo& getNeighborInfo(int newId) const {
        static NodeNeighborInfo empty;
        if (newId < 0 || newId >= (int)nodeNeighborInfo_.size()) return empty;
        return nodeNeighborInfo_[newId];
    }

    // ========== 状态查询 ==========

    bool isBuilt() const { return isBuilt_; }
    bool hasSubgrouping() const { return hasSubgrouping_; }
    int getNumClusters() const { return numClusters_; }

    // ========== 序列化 ==========

    void save(std::ofstream& ofs) const;
    void load(std::ifstream& ifs);

    // ========== 统计信息 ==========

    void printStats() const;

    // ========== 直接访问内部数据（供 HNSWQuantizedIndex 使用）==========
    // 注意：这些方法用于兼容现有代码，逐步迁移后应移除

    const std::vector<int>& getNodeToNewId() const { return nodeToNewId_; }
    const std::vector<int>& getNewIdToNode() const { return newIdToNode_; }
    const std::vector<int>& getClusterOffsets() const { return clusterOffset_; }
    const std::vector<std::vector<int>>& getSubgroupOffsets() const { return subgroupOffset_; }
    const std::vector<int>& getNodeSubgroup() const { return nodeSubgroup_; }
    const std::vector<int>& getNodeLocalIdx() const { return nodeLocalIdx_; }
    const std::vector<NodeNeighborInfo>& getAllNeighborInfo() const { return nodeNeighborInfo_; }

    // Move 版本（build 后 SubgroupManager 不再使用，避免深拷贝）
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

    // 节点重编号映射
    std::vector<int> nodeToNewId_;      // originalId -> newId
    std::vector<int> newIdToNode_;      // newId -> originalId

    // 聚类和子组边界
    std::vector<int> clusterOffset_;    // clusterOffset_[c] = 聚类c的起始新ID
    std::vector<std::vector<int>> subgroupOffset_;  // subgroupOffset_[c][g] = 子组起始偏移

    // 子组映射
    std::vector<int> nodeSubgroup_;     // newId -> subgroupId (within cluster)
    std::vector<int> nodeLocalIdx_;     // newId -> localIdx in subgroup

    // 子组到PIR列的映射
    std::vector<std::vector<int>> subgroupToColumn_;

    // 邻居信息
    std::vector<NodeNeighborInfo> nodeNeighborInfo_;
};

} // namespace hnsw

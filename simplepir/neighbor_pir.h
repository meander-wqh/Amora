/**
 * @file neighbor_pir.h
 * @brief Neighbor PIR - 隐私查询节点邻居信息（子组级编码）
 *
 * 与 embedding_pir.h 完全对称的类体系，模板化 <typename ElemType>。
 *
 * 数据库布局: (maxSubgroupSize × M0) × totalSubgroups 矩阵
 *   - 每列: 一个子组 (c, g)
 *   - 每行: 子组内第 i 个节点的第 j 个邻居 (i = row/M0, j = row%M0)
 *   - 每个元素: 编码值（如 22 位编码: 子组ID 14 位 + 局部索引 8 位）
 *
 * 查询: 选择子组列 col
 *   - q[col] += delta (单位向量缩放)
 *   - 其他位置 = LWE 噪声
 *
 * 结果: (maxSubgroupSize × M0) 个编码邻居值
 */

#ifndef NEIGHBOR_PIR_H
#define NEIGHBOR_PIR_H

#include "pir_types.h"
#include "matrix.h"
#include "random.h"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

namespace simplepir {

// ============================================================================
// 配置（只含 NeighborPIR 独有的结构参数，不含 logQ/lweN/sigma）
// ============================================================================

struct NeighborPIRConfig {
    uint64_t totalSubgroups = 0;        // 总子组数（= PIR 矩阵列数）
    uint64_t maxSubgroupSize = 0;       // 子组最大节点数
    uint64_t maxNeighborsPerNode = 0;   // M0
    uint64_t numParts = 1;              // 拆分编码部分数（每个邻居编码拆为 numParts 个部分）

    NeighborPIRConfig() = default;
    NeighborPIRConfig(uint64_t subgroups, uint64_t maxSize, uint64_t maxNbrs, uint64_t parts = 1)
        : totalSubgroups(subgroups), maxSubgroupSize(maxSize), maxNeighborsPerNode(maxNbrs), numParts(parts) {}

    uint64_t totalRows() const { return maxSubgroupSize * maxNeighborsPerNode * numParts; }
    bool isValid() const {
        return totalSubgroups > 0 && maxSubgroupSize > 0 && maxNeighborsPerNode > 0 && numParts > 0;
    }
    void print() const;
};

struct NeighborPIRParams {
    NeighborPIRConfig config;
    Params pirParams;               // N, Sigma, L, M, Logq, P

    // P 默认 2^22（22位编码），invalidNeighbor = P - 1
    uint32_t invalidNeighbor() const { return static_cast<uint32_t>(pirParams.P - 1); }

    uint64_t delta() const { return pirParams.delta(); }

    void init(const NeighborPIRConfig& cfg,
              uint64_t logQ = 64, uint64_t lweN = 1024, double sigma = 6.4,
              uint64_t P = (1ULL << 22));
};

// ============================================================================
// 模板化消息和状态类型
// ============================================================================

template<typename ElemType>
struct NbrQueryContextT {
    std::shared_ptr<MatrixT<ElemType>> Hs;
    NbrQueryContextT() = default;
    explicit NbrQueryContextT(std::shared_ptr<MatrixT<ElemType>> hs) : Hs(std::move(hs)) {}
};
using NbrQueryContext = NbrQueryContextT<Elem64>;

template<typename ElemType>
struct NbrQueryMsgT {
    std::shared_ptr<MatrixT<ElemType>> queryVector;
    uint64_t batchSize = 1;
};
using NbrQueryMsg = NbrQueryMsgT<Elem64>;

template<typename ElemType>
struct NbrAnswerMsgT {
    std::shared_ptr<MatrixT<ElemType>> answer;
};
using NbrAnswerMsg = NbrAnswerMsgT<Elem64>;

// ============================================================================
// 数据库：(maxSubgroupSize × M0) × totalSubgroups 矩阵
// ============================================================================

template<typename ElemType>
class NeighborDatabaseT {
public:
    NeighborDatabaseT() = default;
    explicit NeighborDatabaseT(const NeighborPIRConfig& config);

    // 设置指定位置的编码值
    void setEncodedNeighbor(int subgroupColumn, int localNodeIdx,
                            int neighborSlot, uint64_t encodedValue);

    // 直接设置矩阵元素（用于拆分编码，直接按行列设置）
    void setElement(int row, int col, ElemType value);

    // 获取数据库矩阵
    std::shared_ptr<MatrixT<ElemType>> getMatrix() const { return data_; }
    const NeighborPIRConfig& getConfig() const { return config_; }
    bool isReady() const { return isInitialized_; }
    void printInfo() const;

    // Elem32 直接压缩存储访问
    std::vector<uint8_t>& compressedDataRef() { return compressedData_; }
    uint64_t getDBRows() const { return dbRows_; }
    uint64_t getDBCols() const { return dbCols_; }
    bool hasCompressedData() const { return !compressedData_.empty(); }

private:
    NeighborPIRConfig config_;
    std::shared_ptr<MatrixT<ElemType>> data_;          // Elem64 路径使用
    std::vector<uint8_t> compressedData_;               // Elem32 路径直接 uint8 存储
    uint64_t dbRows_ = 0, dbCols_ = 0;
    bool isInitialized_ = false;
};
using NeighborDatabase = NeighborDatabaseT<Elem64>;

// ============================================================================
// 服务端
// ============================================================================

template<typename ElemType>
class NeighborPIRServerT {
public:
    NeighborPIRServerT() = default;

    // setup: 传入 db + params + sharedMatrix，返回 hint
    std::shared_ptr<MatrixT<ElemType>> setup(
        NeighborDatabaseT<ElemType>& db,
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix
    );

    // 从缓存加载 hint（只压缩 DB，跳过 hint 计算）
    void setupWithCache(
        NeighborDatabaseT<ElemType>& db,
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& cachedHint
    );

    // answer: DB × query
    NbrAnswerMsgT<ElemType> answer(const NbrQueryMsgT<ElemType>& query) const;

    // 批量应答: DB(L×M) × Q(M×N) = Ans(L×N)
    std::shared_ptr<MatrixT<ElemType>> batchAnswer(
        const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const;

    bool isReady() const { return isReady_; }

private:
    std::shared_ptr<MatrixT<ElemType>> database_;
    std::shared_ptr<MatrixT<ElemType>> hint_;
    NeighborPIRParams params_;
    bool isReady_ = false;

    // 压缩存储: DB 值 ∈ [0,127]，以 uint8_t 紧凑存储
    std::vector<uint8_t> compressedDB_;
    uint64_t compressedDBRows_ = 0;
    uint64_t compressedDBCols_ = 0;
};
using NeighborPIRServer = NeighborPIRServerT<Elem64>;

// ============================================================================
// 客户端
// ============================================================================

template<typename ElemType>
class NeighborPIRClientT {
public:
    NeighborPIRClientT() = default;

    // 初始化
    void init(
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix,
        const std::shared_ptr<MatrixT<ElemType>>& hint
    );

    // 查询指定子组列（每次生成新 secret）
    std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
    query(int subgroupColumn);

    // 查询（使用预计算的 As 和 Hs）
    std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
    query(int subgroupColumn,
          const std::shared_ptr<MatrixT<ElemType>>& precomputedAs,
          const std::shared_ptr<MatrixT<ElemType>>& precomputedHs);

    // 恢复指定节点的 M0 个邻居原始值
    std::vector<uint64_t> recoverNodeValues(
        const NbrAnswerMsgT<ElemType>& answer,
        const NbrQueryContextT<ElemType>& ctx,
        int localNodeIdx
    );

    // Getters
    bool isReady() const { return isInitialized_; }
    std::shared_ptr<MatrixT<ElemType>> getHint() const { return hint_; }
    std::shared_ptr<MatrixT<ElemType>> getSharedMatrix() const { return sharedMatrix_; }
    const NeighborPIRParams& getParams() const { return params_; }

private:
    NeighborPIRParams params_;
    std::shared_ptr<MatrixT<ElemType>> sharedMatrix_;
    std::shared_ptr<MatrixT<ElemType>> hint_;
    bool isInitialized_ = false;
};
using NeighborPIRClient = NeighborPIRClientT<Elem64>;

// ============================================================================
// 主协调类
// ============================================================================

template<typename ElemType>
class NeighborPIRT {
public:
    explicit NeighborPIRT(const NeighborPIRConfig& config);
    const NeighborPIRParams& getParams() const { return params_; }
    std::shared_ptr<MatrixT<ElemType>> generateSharedMatrix();
    void printBandwidth() const;

private:
    NeighborPIRConfig config_;
    NeighborPIRParams params_;
};
using NeighborPIR = NeighborPIRT<Elem64>;

// 显式实例化声明
extern template struct NbrQueryContextT<Elem32>;
extern template struct NbrQueryContextT<Elem64>;
extern template struct NbrQueryMsgT<Elem32>;
extern template struct NbrQueryMsgT<Elem64>;
extern template struct NbrAnswerMsgT<Elem32>;
extern template struct NbrAnswerMsgT<Elem64>;
extern template class NeighborDatabaseT<Elem32>;
extern template class NeighborDatabaseT<Elem64>;
extern template class NeighborPIRServerT<Elem32>;
extern template class NeighborPIRServerT<Elem64>;
extern template class NeighborPIRClientT<Elem32>;
extern template class NeighborPIRClientT<Elem64>;
extern template class NeighborPIRT<Elem32>;
extern template class NeighborPIRT<Elem64>;

} // namespace simplepir

#endif // NEIGHBOR_PIR_H

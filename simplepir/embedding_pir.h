/**
 * @file embedding_pir.h
 * @brief Embedding PIR - Private Information Retrieval for Embedding Inner Products
 *
 * This extends SimplePIR to support querying inner products between a query embedding
 * and all embeddings within a selected cluster, while hiding which cluster is queried.
 *
 * Database Layout: K × (C·d) matrix
 *   - K = max embeddings per cluster
 *   - C = number of clusters
 *   - d = embedding dimension
 *   - DB[k, c*d : (c+1)*d] = cluster c's k-th embedding
 *
 * Query: Select cluster c with query embedding v
 *   - q[c*d : (c+1)*d] = δ·v (scaled query embedding)
 *   - Other positions = LWE noise
 *
 * Result: K inner products (one per embedding in cluster c)
 *
 * 所有核心类均已模板化 <typename ElemType>，支持 32/64 位切换。
 */

#ifndef EMBEDDING_PIR_H
#define EMBEDDING_PIR_H

#include "pir_types.h"
#include "matrix.h"
#include "random.h"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

namespace simplepir {

// ============================================================================
// 配置（只含独有的结构参数，不含 logQ/lweN/sigma）
// ============================================================================

struct EmbeddingPIRConfig {
    uint64_t embeddingDim;       // d: dimension of each embedding
    uint64_t numClusters;        // C: number of clusters
    uint64_t maxClusterSize;     // K: maximum embeddings per cluster

    EmbeddingPIRConfig()
        : embeddingDim(0), numClusters(0), maxClusterSize(0) {}

    EmbeddingPIRConfig(uint64_t dim, uint64_t clusters, uint64_t clusterSize)
        : embeddingDim(dim), numClusters(clusters), maxClusterSize(clusterSize) {}

    bool isValid() const {
        return embeddingDim > 0 && numClusters > 0 && maxClusterSize > 0;
    }

    uint64_t queryDim() const { return numClusters * embeddingDim; }
    uint64_t resultSize() const { return maxClusterSize; }
    void print() const;
};

struct EmbeddingPIRParams {
    EmbeddingPIRConfig config;
    Params pirParams;            // Underlying SimplePIR parameters

    uint64_t K() const { return config.maxClusterSize; }
    uint64_t C() const { return config.numClusters; }
    uint64_t d() const { return config.embeddingDim; }
    uint64_t delta() const { return pirParams.delta(); }

    // logQ/lweN/sigma/P 作为参数传入，只存在 pirParams 中
    void init(const EmbeddingPIRConfig& cfg,
              uint64_t logQ = 32, uint64_t lweN = 1024, double sigma = 6.4,
              uint64_t P = 16777216);
};

// ============================================================================
// 模板化消息和状态类型
// ============================================================================

/**
 * @brief 查询上下文，包含恢复所需的状态
 *
 * 每次PIR查询都使用新的secret s，对应的Hs保存在此结构中用于recover。
 */
template<typename ElemType>
struct QueryContextT {
    std::shared_ptr<MatrixT<ElemType>> Hs;  // H × s，用于recover

    QueryContextT() = default;
    explicit QueryContextT(std::shared_ptr<MatrixT<ElemType>> hs) : Hs(std::move(hs)) {}
};
using QueryContext = QueryContextT<Elem64>;

template<typename ElemType>
struct QueryMsgT {
    std::shared_ptr<MatrixT<ElemType>> queryVector;  // q = A^T·s + e + δ·mask(v)
    uint64_t batchSize = 1;
};
using QueryMsg = QueryMsgT<Elem64>;

template<typename ElemType>
struct AnswerMsgT {
    std::shared_ptr<MatrixT<ElemType>> answer;  // ans = DB · q
};
using AnswerMsg = AnswerMsgT<Elem64>;

template<typename ElemType>
struct ServerStateT {
    std::shared_ptr<MatrixT<ElemType>> hint;    // H = DB · A (precomputed)
    bool isReady = false;
};
using ServerState = ServerStateT<Elem64>;

template<typename ElemType>
struct ClientStateT {
    std::shared_ptr<MatrixT<ElemType>> secret;  // LWE secret s
    uint64_t queriedCluster;
    std::vector<float> queryEmbedding;
    bool isReady = false;
};
using ClientState = ClientStateT<Elem64>;

// ============================================================================
// 数据库：K × (C·d) 矩阵
// ============================================================================

template<typename ElemType>
class EmbeddingDatabaseT {
public:
    EmbeddingDatabaseT() = default;
    explicit EmbeddingDatabaseT(const EmbeddingPIRConfig& config);

    uint64_t addEmbedding(uint64_t clusterId, const std::vector<float>& embedding);
    uint64_t addEmbedding(uint64_t clusterId, const std::vector<uint64_t>& embedding);
    void addEmbeddings(uint64_t clusterId, const std::vector<std::vector<float>>& embeddings);

    // 直接写入 uint8 原始数据（Elem32 路径跳过转换链）
    void addEmbeddingRaw(uint64_t clusterId, const uint8_t* raw, int dim);

    uint64_t getClusterSize(uint64_t clusterId) const;
    std::vector<uint64_t> getEmbedding(uint64_t clusterId, uint64_t embeddingIdx) const;

    bool isReady() const { return isInitialized_ && hasData_; }
    void finalize();

    const EmbeddingPIRConfig& getConfig() const { return config_; }
    std::shared_ptr<MatrixT<ElemType>> getMatrix() const { return data_; }
    void printInfo() const;

    // Elem32 直接压缩存储访问
    std::vector<uint8_t>& compressedDataRef() { return compressedData_; }
    uint64_t getDBRows() const { return dbRows_; }
    uint64_t getDBCols() const { return dbCols_; }
    bool hasCompressedData() const { return !compressedData_.empty(); }

private:
    EmbeddingPIRConfig config_;
    std::shared_ptr<MatrixT<ElemType>> data_;          // Elem64 路径使用
    std::vector<uint8_t> compressedData_;               // Elem32 路径直接 uint8 存储
    uint64_t dbRows_ = 0, dbCols_ = 0;
    std::vector<uint64_t> clusterSizes_;
    bool isInitialized_ = false;
    bool hasData_ = false;
    bool isFinalized_ = false;

    uint64_t quantizeValue(float val) const;
    float dequantizeValue(uint64_t val) const;
};
using EmbeddingDatabase = EmbeddingDatabaseT<Elem64>;

// ============================================================================
// 服务端
// ============================================================================

template<typename ElemType>
class EmbeddingPIRServerT {
public:
    EmbeddingPIRServerT() = default;

    std::shared_ptr<MatrixT<ElemType>> setup(
        EmbeddingDatabaseT<ElemType>& db,
        const EmbeddingPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix
    );

    AnswerMsgT<ElemType> answer(const QueryMsgT<ElemType>& query) const;

    // 批量查询应答: DB(K×M) × Q(M×N) = Ans(K×N)
    // queryMatrix 为 (C·d) × batchSize 矩阵
    std::shared_ptr<MatrixT<ElemType>> batchAnswer(
        const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const;

    // 从缓存加载 hint（只压缩 DB，跳过 hint 计算）
    void setupWithCache(
        EmbeddingDatabaseT<ElemType>& db,
        const EmbeddingPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& cachedHint
    );

    bool isReady() const { return state_.isReady; }

    // 设置有符号数据标志（用于 int8 量化数据如 MS-MARCO）
    void setSignedData(bool signedData) { signedData_ = signedData; }

private:
    ServerStateT<ElemType> state_;
    std::shared_ptr<MatrixT<ElemType>> database_;
    EmbeddingPIRParams params_;
    bool signedData_ = false;  // true 时压缩 matmul 使用 int8 符号扩展

    // 压缩存储: DB 值 ∈ [0,255]，以 uint8_t 紧凑存储，节省 ~4x 内存带宽
    std::vector<uint8_t> compressedDB_;
    uint64_t compressedDBRows_ = 0;
    uint64_t compressedDBCols_ = 0;
};
using EmbeddingPIRServer = EmbeddingPIRServerT<Elem64>;

// ============================================================================
// 客户端
// ============================================================================

template<typename ElemType>
class EmbeddingPIRClientT {
public:
    EmbeddingPIRClientT() = default;

    void init(
        const EmbeddingPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix,
        const std::shared_ptr<MatrixT<ElemType>>& hint
    );

    std::pair<QueryMsgT<ElemType>, QueryContextT<ElemType>>
    query(uint64_t clusterId, const std::vector<float>& queryEmbedding);

    std::pair<QueryMsgT<ElemType>, QueryContextT<ElemType>>
    query(uint64_t clusterId, const std::vector<uint64_t>& queryEmbedding);

    std::vector<int64_t> recover(const AnswerMsgT<ElemType>& answer, const QueryContextT<ElemType>& ctx);
    std::vector<float> recoverFloat(const AnswerMsgT<ElemType>& answer, const QueryContextT<ElemType>& ctx);

    bool isReady() const { return isInitialized_; }
    std::shared_ptr<MatrixT<ElemType>> getHint() const { return hint_; }
    std::shared_ptr<MatrixT<ElemType>> getSharedMatrix() const { return sharedMatrix_; }
    const EmbeddingPIRParams& getParams() const { return params_; }

private:
    EmbeddingPIRParams params_;
    std::shared_ptr<MatrixT<ElemType>> sharedMatrix_;  // A
    std::shared_ptr<MatrixT<ElemType>> hint_;          // H

    bool isInitialized_ = false;

    ClientStateT<ElemType> state_;
};
using EmbeddingPIRClient = EmbeddingPIRClientT<Elem64>;

// ============================================================================
// 主协调类
// ============================================================================

template<typename ElemType>
class EmbeddingPIRT {
public:
    explicit EmbeddingPIRT(const EmbeddingPIRConfig& config);

    const EmbeddingPIRParams& getParams() const { return params_; }

    std::shared_ptr<MatrixT<ElemType>> generateSharedMatrix();
    std::shared_ptr<MatrixT<ElemType>> generateSharedMatrix(const PRGKey& seed);

    void printBandwidth() const;

    static int64_t computeInnerProduct(
        const std::vector<uint64_t>& a,
        const std::vector<uint64_t>& b
    );

    static float computeInnerProductFloat(
        const std::vector<float>& a,
        const std::vector<float>& b
    );

private:
    EmbeddingPIRConfig config_;
    EmbeddingPIRParams params_;
};
using EmbeddingPIR = EmbeddingPIRT<Elem64>;

// 显式实例化声明
extern template struct QueryContextT<Elem32>;
extern template struct QueryContextT<Elem64>;
extern template struct QueryMsgT<Elem32>;
extern template struct QueryMsgT<Elem64>;
extern template struct AnswerMsgT<Elem32>;
extern template struct AnswerMsgT<Elem64>;
extern template struct ServerStateT<Elem32>;
extern template struct ServerStateT<Elem64>;
extern template struct ClientStateT<Elem32>;
extern template struct ClientStateT<Elem64>;
extern template class EmbeddingDatabaseT<Elem32>;
extern template class EmbeddingDatabaseT<Elem64>;
extern template class EmbeddingPIRServerT<Elem32>;
extern template class EmbeddingPIRServerT<Elem64>;
extern template class EmbeddingPIRClientT<Elem32>;
extern template class EmbeddingPIRClientT<Elem64>;
extern template class EmbeddingPIRT<Elem32>;
extern template class EmbeddingPIRT<Elem64>;

} // namespace simplepir

#endif // EMBEDDING_PIR_H

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

struct EmbeddingPIRConfig {
    uint64_t embeddingDim;
    uint64_t numClusters;
    uint64_t maxClusterSize;

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
    Params pirParams;

    uint64_t K() const { return config.maxClusterSize; }
    uint64_t C() const { return config.numClusters; }
    uint64_t d() const { return config.embeddingDim; }
    uint64_t delta() const { return pirParams.delta(); }

    void init(const EmbeddingPIRConfig& cfg,
              uint64_t logQ = 32, uint64_t lweN = 1024, double sigma = 6.4,
              uint64_t P = 16777216);
};

template<typename ElemType>
struct QueryContextT {
    std::shared_ptr<MatrixT<ElemType>> Hs;

    QueryContextT() = default;
    explicit QueryContextT(std::shared_ptr<MatrixT<ElemType>> hs) : Hs(std::move(hs)) {}
};
using QueryContext = QueryContextT<Elem64>;

template<typename ElemType>
struct QueryMsgT {
    std::shared_ptr<MatrixT<ElemType>> queryVector;
    uint64_t batchSize = 1;
};
using QueryMsg = QueryMsgT<Elem64>;

template<typename ElemType>
struct AnswerMsgT {
    std::shared_ptr<MatrixT<ElemType>> answer;
};
using AnswerMsg = AnswerMsgT<Elem64>;

template<typename ElemType>
struct ServerStateT {
    std::shared_ptr<MatrixT<ElemType>> hint;
    bool isReady = false;
};
using ServerState = ServerStateT<Elem64>;

template<typename ElemType>
struct ClientStateT {
    std::shared_ptr<MatrixT<ElemType>> secret;
    uint64_t queriedCluster;
    std::vector<float> queryEmbedding;
    bool isReady = false;
};
using ClientState = ClientStateT<Elem64>;

template<typename ElemType>
class EmbeddingDatabaseT {
public:
    EmbeddingDatabaseT() = default;
    explicit EmbeddingDatabaseT(const EmbeddingPIRConfig& config);

    uint64_t addEmbedding(uint64_t clusterId, const std::vector<float>& embedding);
    uint64_t addEmbedding(uint64_t clusterId, const std::vector<uint64_t>& embedding);
    void addEmbeddings(uint64_t clusterId, const std::vector<std::vector<float>>& embeddings);

    void addEmbeddingRaw(uint64_t clusterId, const uint8_t* raw, int dim);

    uint64_t getClusterSize(uint64_t clusterId) const;
    std::vector<uint64_t> getEmbedding(uint64_t clusterId, uint64_t embeddingIdx) const;

    bool isReady() const { return isInitialized_ && hasData_; }
    void finalize();

    const EmbeddingPIRConfig& getConfig() const { return config_; }
    std::shared_ptr<MatrixT<ElemType>> getMatrix() const { return data_; }
    void printInfo() const;

    std::vector<uint8_t>& compressedDataRef() { return compressedData_; }
    uint64_t getDBRows() const { return dbRows_; }
    uint64_t getDBCols() const { return dbCols_; }
    bool hasCompressedData() const { return !compressedData_.empty(); }

private:
    EmbeddingPIRConfig config_;
    std::shared_ptr<MatrixT<ElemType>> data_;
    std::vector<uint8_t> compressedData_;
    uint64_t dbRows_ = 0, dbCols_ = 0;
    std::vector<uint64_t> clusterSizes_;
    bool isInitialized_ = false;
    bool hasData_ = false;
    bool isFinalized_ = false;

    uint64_t quantizeValue(float val) const;
    float dequantizeValue(uint64_t val) const;
};
using EmbeddingDatabase = EmbeddingDatabaseT<Elem64>;

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

    std::shared_ptr<MatrixT<ElemType>> batchAnswer(
        const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const;

    void setupWithCache(
        EmbeddingDatabaseT<ElemType>& db,
        const EmbeddingPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& cachedHint
    );

    bool isReady() const { return state_.isReady; }

    void setSignedData(bool signedData) { signedData_ = signedData; }

private:
    ServerStateT<ElemType> state_;
    std::shared_ptr<MatrixT<ElemType>> database_;
    EmbeddingPIRParams params_;
    bool signedData_ = false;

    std::vector<uint8_t> compressedDB_;
    uint64_t compressedDBRows_ = 0;
    uint64_t compressedDBCols_ = 0;
};
using EmbeddingPIRServer = EmbeddingPIRServerT<Elem64>;

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
    std::shared_ptr<MatrixT<ElemType>> sharedMatrix_;
    std::shared_ptr<MatrixT<ElemType>> hint_;

    bool isInitialized_ = false;

    ClientStateT<ElemType> state_;
};
using EmbeddingPIRClient = EmbeddingPIRClientT<Elem64>;

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

}

#endif

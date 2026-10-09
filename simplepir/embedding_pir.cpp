#include "embedding_pir.h"
#include "pir_math.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace simplepir {

void EmbeddingPIRConfig::print() const {
    std::cout << "=== Embedding PIR Configuration ===" << std::endl;
    std::cout << "Embedding dimension (d): " << embeddingDim << std::endl;
    std::cout << "Number of clusters (C): " << numClusters << std::endl;
    std::cout << "Max cluster size (K):   " << maxClusterSize << std::endl;
    std::cout << "Query vector dimension: " << queryDim() << std::endl;
    std::cout << "Result size:            " << resultSize() << std::endl;
    std::cout << "===================================" << std::endl;
}

void EmbeddingPIRParams::init(const EmbeddingPIRConfig& cfg,
                               uint64_t logQ, uint64_t lweN, double sigma,
                               uint64_t P) {
    config = cfg;

    pirParams.N = lweN;
    pirParams.Sigma = sigma;
    pirParams.L = cfg.maxClusterSize;
    pirParams.M = cfg.numClusters * cfg.embeddingDim;
    pirParams.Logq = logQ;
    pirParams.P = P;
}

template<typename ElemType>
EmbeddingDatabaseT<ElemType>::EmbeddingDatabaseT(const EmbeddingPIRConfig& config)
    : config_(config), isInitialized_(true), hasData_(false), isFinalized_(false) {

    if (!config.isValid()) {
        throw std::invalid_argument("Invalid EmbeddingPIRConfig");
    }

    clusterSizes_.resize(config.numClusters, 0);

    uint64_t rows = config.maxClusterSize;
    uint64_t cols = config.numClusters * config.embeddingDim;

    if constexpr (sizeof(ElemType) == 4) {
        dbRows_ = rows;
        dbCols_ = cols;
        compressedData_.resize(rows * cols, 0);
    } else {
        data_ = MatrixT<ElemType>::zeros(rows, cols);
    }
}

template<typename ElemType>
uint64_t EmbeddingDatabaseT<ElemType>::addEmbedding(uint64_t clusterId, const std::vector<float>& embedding) {
    std::vector<uint64_t> quantized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        quantized[i] = quantizeValue(embedding[i]);
    }
    return addEmbedding(clusterId, quantized);
}

template<typename ElemType>
uint64_t EmbeddingDatabaseT<ElemType>::addEmbedding(uint64_t clusterId, const std::vector<uint64_t>& embedding) {
    if (!isInitialized_) {
        throw std::runtime_error("Database not initialized");
    }
    if (clusterId >= config_.numClusters) {
        throw std::out_of_range("Cluster ID out of range");
    }
    if (embedding.size() != config_.embeddingDim) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }
    if (clusterSizes_[clusterId] >= config_.maxClusterSize) {
        throw std::overflow_error("Cluster is full");
    }

    uint64_t embIdx = clusterSizes_[clusterId];
    uint64_t colOffset = clusterId * config_.embeddingDim;

    if constexpr (sizeof(ElemType) == 4) {
        for (uint64_t i = 0; i < config_.embeddingDim; ++i) {
            compressedData_[embIdx * dbCols_ + colOffset + i] = static_cast<uint8_t>(embedding[i]);
        }
    } else {
        for (uint64_t i = 0; i < config_.embeddingDim; ++i) {
            data_->set(embIdx, colOffset + i, embedding[i]);
        }
    }

    clusterSizes_[clusterId]++;
    hasData_ = true;

    return embIdx;
}

template<typename ElemType>
void EmbeddingDatabaseT<ElemType>::addEmbeddingRaw(uint64_t clusterId, const uint8_t* raw, int dim) {
    if (clusterId >= config_.numClusters) {
        throw std::out_of_range("Cluster ID out of range");
    }
    if (clusterSizes_[clusterId] >= config_.maxClusterSize) {
        throw std::overflow_error("Cluster is full");
    }

    uint64_t embIdx = clusterSizes_[clusterId];
    uint64_t colOffset = clusterId * config_.embeddingDim;

    if constexpr (sizeof(ElemType) == 4) {
        std::memcpy(&compressedData_[embIdx * dbCols_ + colOffset], raw, dim);
    } else {
        for (int i = 0; i < dim; ++i) {
            data_->set(embIdx, colOffset + i, static_cast<ElemType>(raw[i]));
        }
    }

    clusterSizes_[clusterId]++;
    hasData_ = true;
}

template<typename ElemType>
void EmbeddingDatabaseT<ElemType>::addEmbeddings(uint64_t clusterId, const std::vector<std::vector<float>>& embeddings) {
    for (const auto& emb : embeddings) {
        addEmbedding(clusterId, emb);
    }
}

template<typename ElemType>
uint64_t EmbeddingDatabaseT<ElemType>::getClusterSize(uint64_t clusterId) const {
    if (clusterId >= config_.numClusters) {
        throw std::out_of_range("Cluster ID out of range");
    }
    return clusterSizes_[clusterId];
}

template<typename ElemType>
std::vector<uint64_t> EmbeddingDatabaseT<ElemType>::getEmbedding(uint64_t clusterId, uint64_t embeddingIdx) const {
    if (clusterId >= config_.numClusters) {
        throw std::out_of_range("Cluster ID out of range");
    }
    if (embeddingIdx >= clusterSizes_[clusterId]) {
        throw std::out_of_range("Embedding index out of range");
    }

    std::vector<uint64_t> result(config_.embeddingDim);
    uint64_t colOffset = clusterId * config_.embeddingDim;

    if constexpr (sizeof(ElemType) == 4) {
        for (uint64_t i = 0; i < config_.embeddingDim; ++i) {
            result[i] = compressedData_[embeddingIdx * dbCols_ + colOffset + i];
        }
    } else {
        for (uint64_t i = 0; i < config_.embeddingDim; ++i) {
            result[i] = data_->get(embeddingIdx, colOffset + i);
        }
    }

    return result;
}

template<typename ElemType>
void EmbeddingDatabaseT<ElemType>::finalize() {
    if (!hasData_) {
        throw std::runtime_error("No data in database");
    }
    isFinalized_ = true;
}

template<typename ElemType>
void EmbeddingDatabaseT<ElemType>::printInfo() const {
    std::cout << "=== Embedding Database Info ===" << std::endl;
    std::cout << "Dimensions: " << config_.embeddingDim << std::endl;
    std::cout << "Clusters: " << config_.numClusters << std::endl;
    std::cout << "Max per cluster: " << config_.maxClusterSize << std::endl;
    if constexpr (sizeof(ElemType) == 4) {
        std::cout << "Matrix size: " << dbRows_ << " x " << dbCols_ << " (compressed uint8)" << std::endl;
    } else {
        std::cout << "Matrix size: " << data_->rows << " x " << data_->cols << std::endl;
    }

    uint64_t totalEmbeddings = 0;
    for (uint64_t i = 0; i < config_.numClusters; ++i) {
        totalEmbeddings += clusterSizes_[i];
    }
    std::cout << "Total embeddings: " << totalEmbeddings << std::endl;

    std::cout << "Cluster sizes: [";
    for (uint64_t i = 0; i < std::min(config_.numClusters, (uint64_t)10); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << clusterSizes_[i];
    }
    if (config_.numClusters > 10) std::cout << ", ...";
    std::cout << "]" << std::endl;
    std::cout << "===============================" << std::endl;
}

template<typename ElemType>
uint64_t EmbeddingDatabaseT<ElemType>::quantizeValue(float val) const {
    float scaled = (val + 1.0f) * 7.5f;
    scaled = std::max(0.0f, std::min(15.0f, scaled));
    return static_cast<uint64_t>(scaled);
}

template<typename ElemType>
float EmbeddingDatabaseT<ElemType>::dequantizeValue(uint64_t val) const {
    return static_cast<float>(val) / 7.5f - 1.0f;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> EmbeddingPIRServerT<ElemType>::setup(
    EmbeddingDatabaseT<ElemType>& db,
    const EmbeddingPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix
) {
    params_ = params;

    if constexpr (sizeof(ElemType) == 4) {
        if (db.hasCompressedData()) {
            compressedDBRows_ = db.getDBRows();
            compressedDBCols_ = db.getDBCols();
            compressedDB_ = std::move(db.compressedDataRef());

            double compMB = (double)compressedDB_.size() / (1024 * 1024);
            std::cout << "[EmbeddingPIRServer] DB direct uint8: " << std::fixed << std::setprecision(2)
                      << compMB << " MB (zero-copy from database)" << std::endl;
        } else {
            database_ = db.getMatrix();
            compressedDBRows_ = database_->rows;
            compressedDBCols_ = database_->cols;
            compressedDB_.resize(compressedDBRows_ * compressedDBCols_);
            const auto* src = database_->data.data();
            for (size_t i = 0; i < compressedDBRows_ * compressedDBCols_; i++) {
                compressedDB_[i] = static_cast<uint8_t>(src[i]);
            }
            database_.reset();
        }

        state_.hint = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, sharedMatrix->cols);
        matMulCompressed8_32(state_.hint->data.data(), compressedDB_.data(),
                              sharedMatrix->data.data(),
                              compressedDBRows_, compressedDBCols_, sharedMatrix->cols, signedData_);
    } else {
        database_ = db.getMatrix();
        compressedDBRows_ = database_->rows;
        compressedDBCols_ = database_->cols;
        compressedDB_.resize(compressedDBRows_ * compressedDBCols_);
        const auto* src = database_->data.data();
        for (size_t i = 0; i < compressedDBRows_ * compressedDBCols_; i++) {
            compressedDB_[i] = static_cast<uint8_t>(src[i]);
        }
        state_.hint = matrixMul(database_, sharedMatrix);
        database_.reset();
    }

    state_.isReady = true;

    std::cout << "[EmbeddingPIRServer] Setup complete. Hint size: "
              << state_.hint->rows << " x " << state_.hint->cols << std::endl;

    return state_.hint;
}

template<typename ElemType>
void EmbeddingPIRServerT<ElemType>::setupWithCache(
    EmbeddingDatabaseT<ElemType>& db,
    const EmbeddingPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& cachedHint
) {
    params_ = params;
    state_.hint = cachedHint;

    if constexpr (sizeof(ElemType) == 4) {
        if (db.hasCompressedData()) {
            compressedDBRows_ = db.getDBRows();
            compressedDBCols_ = db.getDBCols();
            compressedDB_ = std::move(db.compressedDataRef());
        } else {
            database_ = db.getMatrix();
            compressedDBRows_ = database_->rows;
            compressedDBCols_ = database_->cols;
            compressedDB_.resize(compressedDBRows_ * compressedDBCols_);
            const auto* src = database_->data.data();
            for (size_t i = 0; i < compressedDBRows_ * compressedDBCols_; i++) {
                compressedDB_[i] = static_cast<uint8_t>(src[i]);
            }
            database_.reset();
        }
    } else {
        database_ = db.getMatrix();
        compressedDBRows_ = database_->rows;
        compressedDBCols_ = database_->cols;
        compressedDB_.resize(compressedDBRows_ * compressedDBCols_);
        const auto* src = database_->data.data();
        for (size_t i = 0; i < compressedDBRows_ * compressedDBCols_; i++) {
            compressedDB_[i] = static_cast<uint8_t>(src[i]);
        }
        database_.reset();
    }

    double compMB = (double)compressedDB_.size() / (1024 * 1024);
    std::cout << "[EmbeddingPIRServer] DB compressed (cache mode): "
              << std::fixed << std::setprecision(2) << compMB << " MB" << std::endl;

    state_.isReady = true;
}

template<typename ElemType>
AnswerMsgT<ElemType> EmbeddingPIRServerT<ElemType>::answer(const QueryMsgT<ElemType>& query) const {
    if (!state_.isReady) {
        throw std::runtime_error("Server not initialized");
    }

    AnswerMsgT<ElemType> response;
    if constexpr (sizeof(ElemType) == 4) {
        auto out = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, 1);
        matMulVecCompressed8_32(out->data.data(), compressedDB_.data(),
                                 query.queryVector->data.data(),
                                 compressedDBRows_, compressedDBCols_, signedData_);
        response.answer = out;
    } else {
        response.answer = matrixMulVec(database_, query.queryVector);
    }

    return response;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> EmbeddingPIRServerT<ElemType>::batchAnswer(
    const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const {
    if (!state_.isReady) {
        throw std::runtime_error("Server not initialized");
    }
    if constexpr (sizeof(ElemType) == 4) {
        auto out = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, queryMatrix->cols);
        if (queryMatrix->cols == 1) {
            matMulVecCompressed8_32(out->data.data(), compressedDB_.data(),
                                     queryMatrix->data.data(),
                                     compressedDBRows_, compressedDBCols_, signedData_);
        } else {
            matMulCompressed8_32(out->data.data(), compressedDB_.data(),
                                  queryMatrix->data.data(),
                                  compressedDBRows_, compressedDBCols_, queryMatrix->cols, signedData_);
        }
        return out;
    } else {
        if (queryMatrix->cols == 1) {
            return matrixMulVec(*database_, *queryMatrix);
        }
        return matrixMul(*database_, *queryMatrix);
    }
}

template<typename ElemType>
void EmbeddingPIRClientT<ElemType>::init(
    const EmbeddingPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix,
    const std::shared_ptr<MatrixT<ElemType>>& hint
) {
    params_ = params;
    sharedMatrix_ = sharedMatrix;
    hint_ = hint;
    isInitialized_ = true;

    std::cout << "[EmbeddingPIRClient] Initialized. Hint size: "
              << hint_->rows << " x " << hint_->cols << std::endl;
}

template<typename ElemType>
std::pair<QueryMsgT<ElemType>, QueryContextT<ElemType>>
EmbeddingPIRClientT<ElemType>::query(uint64_t clusterId, const std::vector<float>& queryEmbedding) {
    std::vector<uint64_t> quantized(queryEmbedding.size());
    for (size_t i = 0; i < queryEmbedding.size(); ++i) {
        float scaled = (queryEmbedding[i] + 1.0f) * 7.5f;
        scaled = std::max(0.0f, std::min(15.0f, scaled));
        quantized[i] = static_cast<uint64_t>(scaled);
    }

    state_.queryEmbedding = queryEmbedding;

    return query(clusterId, quantized);
}

template<typename ElemType>
std::pair<QueryMsgT<ElemType>, QueryContextT<ElemType>>
EmbeddingPIRClientT<ElemType>::query(uint64_t clusterId, const std::vector<uint64_t>& queryEmbedding) {
    if (!isInitialized_) {
        throw std::runtime_error("Client not initialized");
    }
    if (clusterId >= params_.C()) {
        throw std::out_of_range("Cluster ID out of range");
    }
    if (queryEmbedding.size() != params_.d()) {
        throw std::invalid_argument("Query embedding dimension mismatch");
    }

    uint64_t N = params_.pirParams.N;
    uint64_t queryDim = params_.config.queryDim();

    auto secret = MatrixT<ElemType>::random(N, 1, params_.pirParams.Logq, 0);

    auto Hs = matrixMulVec(hint_, secret);

    auto queryVec = matrixMulVec(sharedMatrix_, secret);

    auto noise = MatrixT<ElemType>::gaussian(queryDim, 1);
    queryVec->matrixAdd(*noise);

    uint64_t delta = params_.delta();
    uint64_t colOffset = clusterId * params_.d();

    for (uint64_t i = 0; i < params_.d(); ++i) {
        uint64_t currentVal = queryVec->get(colOffset + i, 0);
        uint64_t addition = delta * queryEmbedding[i];
        queryVec->set(colOffset + i, 0, currentVal + addition);
    }

    QueryMsgT<ElemType> msg;
    msg.queryVector = queryVec;
    msg.batchSize = 1;

    QueryContextT<ElemType> ctx(Hs);

    return {msg, ctx};
}

template<typename ElemType>
std::vector<int64_t> EmbeddingPIRClientT<ElemType>::recover(
    const AnswerMsgT<ElemType>& answer, const QueryContextT<ElemType>& ctx
) {
    if (!ctx.Hs) {
        throw std::invalid_argument("QueryContext Hs cannot be null");
    }
    if (!answer.answer) {
        throw std::invalid_argument("Answer cannot be null");
    }

    uint64_t K = params_.K();
    std::vector<int64_t> results(K);

    uint64_t delta = params_.delta();
    uint64_t q = 1ULL << params_.pirParams.Logq;

    const ElemType* ansData = answer.answer->data.data();
    const ElemType* hsData = ctx.Hs->data.data();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (uint64_t i = 0; i < K; ++i) {
        uint64_t ansVal = ansData[i];
        uint64_t hsVal = hsData[i];

        uint64_t diff;
        if (ansVal >= hsVal) {
            diff = ansVal - hsVal;
        } else {
            diff = q - (hsVal - ansVal);
        }

        int64_t signedDiff;
        if (diff > q / 2) {
            signedDiff = static_cast<int64_t>(diff) - static_cast<int64_t>(q);
        } else {
            signedDiff = static_cast<int64_t>(diff);
        }

        int64_t rounded = (signedDiff + static_cast<int64_t>(delta / 2)) / static_cast<int64_t>(delta);

        results[i] = rounded;
    }

    return results;
}

template<typename ElemType>
std::vector<float> EmbeddingPIRClientT<ElemType>::recoverFloat(
    const AnswerMsgT<ElemType>& answer, const QueryContextT<ElemType>& ctx
) {
    auto intResults = recover(answer, ctx);
    std::vector<float> floatResults(intResults.size());

    for (size_t i = 0; i < intResults.size(); ++i) {
        float maxVal = 255.0f * 255.0f * params_.d();
        floatResults[i] = static_cast<float>(intResults[i]) / maxVal;
    }

    return floatResults;
}

template<typename ElemType>
EmbeddingPIRT<ElemType>::EmbeddingPIRT(const EmbeddingPIRConfig& config) : config_(config) {
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid EmbeddingPIRConfig");
    }
    params_.init(config);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> EmbeddingPIRT<ElemType>::generateSharedMatrix() {
    uint64_t rows = config_.queryDim();
    uint64_t cols = params_.pirParams.N;
    return MatrixT<ElemType>::random(rows, cols, params_.pirParams.Logq, 0);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> EmbeddingPIRT<ElemType>::generateSharedMatrix(const PRGKey& seed) {
    uint64_t rows = config_.queryDim();
    uint64_t cols = params_.pirParams.N;

    auto matrix = std::make_shared<MatrixT<ElemType>>(rows, cols);

    PRGReader prg(seed);
    uint64_t mod = 1ULL << params_.pirParams.Logq;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        matrix->data[i] = static_cast<ElemType>(prg.randMod(mod));
    }

    return matrix;
}

template<typename ElemType>
void EmbeddingPIRT<ElemType>::printBandwidth() const {
    uint64_t logq = params_.pirParams.Logq;
    uint64_t N = params_.pirParams.N;
    uint64_t K = params_.K();
    uint64_t Cd = config_.queryDim();

    uint64_t hintBits = K * N * logq;
    uint64_t queryBits = Cd * logq;
    uint64_t answerBits = K * logq;

    std::cout << "=== Bandwidth Analysis ===" << std::endl;
    std::cout << "Offline download (Hint H): " << hintBits / 8 / 1024.0 << " KB" << std::endl;
    std::cout << "Online upload (Query):     " << queryBits / 8 / 1024.0 << " KB" << std::endl;
    std::cout << "Online download (Answer):  " << answerBits / 8 / 1024.0 << " KB" << std::endl;
    std::cout << "===========================" << std::endl;
}

template<typename ElemType>
int64_t EmbeddingPIRT<ElemType>::computeInnerProduct(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector size mismatch");
    }

    int64_t result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    }
    return result;
}

template<typename ElemType>
float EmbeddingPIRT<ElemType>::computeInnerProductFloat(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector size mismatch");
    }

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template struct QueryContextT<Elem32>;
template struct QueryContextT<Elem64>;
template struct QueryMsgT<Elem32>;
template struct QueryMsgT<Elem64>;
template struct AnswerMsgT<Elem32>;
template struct AnswerMsgT<Elem64>;
template struct ServerStateT<Elem32>;
template struct ServerStateT<Elem64>;
template struct ClientStateT<Elem32>;
template struct ClientStateT<Elem64>;
template class EmbeddingDatabaseT<Elem32>;
template class EmbeddingDatabaseT<Elem64>;
template class EmbeddingPIRServerT<Elem32>;
template class EmbeddingPIRServerT<Elem64>;
template class EmbeddingPIRClientT<Elem32>;
template class EmbeddingPIRClientT<Elem64>;
template class EmbeddingPIRT<Elem32>;
template class EmbeddingPIRT<Elem64>;

}

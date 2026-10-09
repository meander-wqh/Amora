#include "neighbor_pir.h"
#include "pir_math.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

namespace simplepir {

namespace {

inline int64_t branchlessClamp(int64_t val, int64_t maxVal) {
    int64_t negMask = val >> 63;
    val = val & ~negMask;

    int64_t diff = maxVal - val;
    int64_t overMask = diff >> 63;
    val = (val & ~overMask) | (maxVal & overMask);

    return val;
}

inline uint64_t branchlessPIRRecover32(
    uint32_t ansVal,
    uint32_t hsVal,
    int64_t halfQMod,
    int64_t signedQMod,
    int64_t halfDelta,
    int64_t signedDelta,
    int64_t signedP
) {
    int64_t diff = static_cast<int64_t>(ansVal) - static_cast<int64_t>(hsVal);

    int64_t negMask = diff >> 63;
    diff += signedQMod & negMask;

    int64_t wrapMask = (halfQMod - diff) >> 63;
    diff -= signedQMod & wrapMask;

    int64_t rounded = (diff + halfDelta) / signedDelta;

    return static_cast<uint64_t>(branchlessClamp(rounded, signedP - 1));
}

inline uint64_t branchlessPIRRecover64(
    uint64_t ansVal,
    uint64_t hsVal,
    int64_t halfDelta,
    int64_t signedDelta,
    int64_t signedP
) {
    int64_t diff = static_cast<int64_t>(ansVal - hsVal);

    int64_t rounded = (diff + halfDelta) / signedDelta;

    return static_cast<uint64_t>(branchlessClamp(rounded, signedP - 1));
}

template<typename ElemType>
inline uint64_t recoverSingleValue(
    ElemType ansVal, ElemType hsVal,
    const Params& pirParams
);

template<>
inline uint64_t recoverSingleValue<Elem64>(
    Elem64 ansVal, Elem64 hsVal,
    const Params& pirParams
) {
    uint64_t delta = pirParams.delta();
    int64_t halfDelta = static_cast<int64_t>(delta / 2);
    int64_t signedDelta = static_cast<int64_t>(delta);
    int64_t signedP = static_cast<int64_t>(pirParams.P);

    return branchlessPIRRecover64(ansVal, hsVal, halfDelta, signedDelta, signedP);
}

template<>
inline uint64_t recoverSingleValue<Elem32>(
    Elem32 ansVal, Elem32 hsVal,
    const Params& pirParams
) {
    uint64_t delta = pirParams.delta();
    int64_t halfDelta = static_cast<int64_t>(delta / 2);
    int64_t signedDelta = static_cast<int64_t>(delta);
    int64_t signedP = static_cast<int64_t>(pirParams.P);

    uint64_t q = 1ULL << pirParams.Logq;
    int64_t halfQMod = static_cast<int64_t>(q / 2);
    int64_t signedQMod = static_cast<int64_t>(q);

    return branchlessPIRRecover32(ansVal, hsVal, halfQMod, signedQMod,
                                   halfDelta, signedDelta, signedP);
}

template<typename ElemType>
inline uint64_t recoverUnsignedValue(
    ElemType ansVal, ElemType hsVal,
    const Params& pirParams
) {
    ElemType diff = ansVal - hsVal;
    uint64_t delta = pirParams.delta();
    uint64_t P = pirParams.P;

    uint64_t diff64 = static_cast<uint64_t>(diff);
    uint64_t rounded = (diff64 + delta / 2) / delta;
    return rounded % P;
}

}

void NeighborPIRConfig::print() const {
    std::cout << "=== Neighbor PIR Configuration ===" << std::endl;
    std::cout << "Total subgroups:        " << totalSubgroups << std::endl;
    std::cout << "Max subgroup size:      " << maxSubgroupSize << std::endl;
    std::cout << "Max neighbors per node: " << maxNeighborsPerNode << std::endl;
    std::cout << "Num parts (split enc):  " << numParts << std::endl;
    std::cout << "Total rows (L):         " << totalRows() << std::endl;
    std::cout << "==================================" << std::endl;
}

void NeighborPIRParams::init(const NeighborPIRConfig& cfg,
                              uint64_t logQ, uint64_t lweN, double sigma,
                              uint64_t P) {
    config = cfg;

    pirParams.N = lweN;
    pirParams.Sigma = sigma;
    pirParams.L = cfg.totalRows();
    pirParams.M = cfg.totalSubgroups;
    pirParams.Logq = logQ;
    pirParams.P = P;
}

template<typename ElemType>
NeighborDatabaseT<ElemType>::NeighborDatabaseT(const NeighborPIRConfig& config)
    : config_(config), isInitialized_(false) {

    if (!config.isValid()) {
        throw std::invalid_argument("Invalid NeighborPIRConfig");
    }

    uint64_t rows = config.totalRows();
    uint64_t cols = config.totalSubgroups;

    if constexpr (sizeof(ElemType) == 4) {
        dbRows_ = rows;
        dbCols_ = cols;
        compressedData_.resize(rows * cols, 127);
    } else {
        data_ = std::make_shared<MatrixT<ElemType>>(rows, cols);
        for (uint64_t r = 0; r < rows; ++r) {
            for (uint64_t c = 0; c < cols; ++c) {
                data_->set(r, c, static_cast<ElemType>(0));
            }
        }
    }

    isInitialized_ = true;
}

template<typename ElemType>
void NeighborDatabaseT<ElemType>::setElement(int row, int col, ElemType value) {
    if constexpr (sizeof(ElemType) == 4) {
        compressedData_[static_cast<uint64_t>(row) * dbCols_ + col] = static_cast<uint8_t>(value);
    } else {
        if (data_) data_->set(row, col, value);
    }
}

template<typename ElemType>
void NeighborDatabaseT<ElemType>::setEncodedNeighbor(
    int subgroupColumn, int localNodeIdx,
    int neighborSlot, uint64_t encodedValue
) {
    if (!isInitialized_) {
        throw std::runtime_error("NeighborDatabase not initialized");
    }

    int M0 = static_cast<int>(config_.maxNeighborsPerNode);
    int nParts = static_cast<int>(config_.numParts);
    int rowIdx = localNodeIdx * M0 * nParts + neighborSlot * nParts;

    if constexpr (sizeof(ElemType) == 4) {
        compressedData_[static_cast<uint64_t>(rowIdx) * dbCols_ + subgroupColumn] = static_cast<uint8_t>(encodedValue);
    } else {
        data_->set(rowIdx, subgroupColumn, static_cast<ElemType>(encodedValue));
    }
}

template<typename ElemType>
void NeighborDatabaseT<ElemType>::printInfo() const {
    std::cout << "=== Neighbor Database Info ===" << std::endl;
    std::cout << "Total subgroups (cols): " << config_.totalSubgroups << std::endl;
    std::cout << "Max subgroup size:      " << config_.maxSubgroupSize << std::endl;
    std::cout << "Max neighbors (M0):     " << config_.maxNeighborsPerNode << std::endl;
    if constexpr (sizeof(ElemType) == 4) {
        if (!compressedData_.empty()) {
            std::cout << "Matrix size: " << dbRows_ << " x " << dbCols_ << " (compressed uint8)" << std::endl;
            double sizeMB = static_cast<double>(compressedData_.size()) / (1024 * 1024);
            std::cout << "Matrix memory: " << std::fixed << std::setprecision(2) << sizeMB << " MB" << std::endl;
        }
    } else {
        if (data_) {
            std::cout << "Matrix size: " << data_->rows << " x " << data_->cols << std::endl;
            double sizeMB = static_cast<double>(data_->sizeBytes()) / (1024 * 1024);
            std::cout << "Matrix memory: " << std::fixed << std::setprecision(2) << sizeMB << " MB" << std::endl;
        }
    }
    std::cout << "==============================" << std::endl;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> NeighborPIRServerT<ElemType>::setup(
    NeighborDatabaseT<ElemType>& db,
    const NeighborPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix
) {
    params_ = params;

    if constexpr (sizeof(ElemType) == 4) {
        if (db.hasCompressedData()) {
            compressedDBRows_ = db.getDBRows();
            compressedDBCols_ = db.getDBCols();
            compressedDB_ = std::move(db.compressedDataRef());

            double compMB = (double)compressedDB_.size() / (1024 * 1024);
            std::cout << "[NeighborPIRServer] DB direct uint8: " << std::fixed << std::setprecision(2)
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

        hint_ = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, sharedMatrix->cols);
        matMulCompressed8_32(hint_->data.data(), compressedDB_.data(),
                              sharedMatrix->data.data(),
                              compressedDBRows_, compressedDBCols_, sharedMatrix->cols, false);
    } else {
        database_ = db.getMatrix();
        compressedDBRows_ = database_->rows;
        compressedDBCols_ = database_->cols;
        compressedDB_.resize(compressedDBRows_ * compressedDBCols_);
        const auto* src = database_->data.data();
        for (size_t i = 0; i < compressedDBRows_ * compressedDBCols_; i++) {
            compressedDB_[i] = static_cast<uint8_t>(src[i]);
        }
        hint_ = matrixMul(database_, sharedMatrix);
        database_.reset();
    }

    isReady_ = true;

    std::cout << "[NeighborPIRServer] Setup complete. Hint: "
              << hint_->rows << " x " << hint_->cols << std::endl;

    return hint_;
}

template<typename ElemType>
void NeighborPIRServerT<ElemType>::setupWithCache(
    NeighborDatabaseT<ElemType>& db,
    const NeighborPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& cachedHint
) {
    params_ = params;
    hint_ = cachedHint;

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
    std::cout << "[NeighborPIRServer] DB compressed (cache mode): "
              << std::fixed << std::setprecision(2) << compMB << " MB" << std::endl;

    isReady_ = true;
}

template<typename ElemType>
NbrAnswerMsgT<ElemType> NeighborPIRServerT<ElemType>::answer(
    const NbrQueryMsgT<ElemType>& query
) const {
    if (!isReady_) {
        throw std::runtime_error("NeighborPIRServer not initialized");
    }

    NbrAnswerMsgT<ElemType> response;
    if constexpr (sizeof(ElemType) == 4) {
        auto out = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, 1);
        matMulVecCompressed8_32(out->data.data(), compressedDB_.data(),
                                 query.queryVector->data.data(),
                                 compressedDBRows_, compressedDBCols_);
        response.answer = out;
    } else {
        response.answer = matrixMulVec(database_, query.queryVector);
    }
    return response;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> NeighborPIRServerT<ElemType>::batchAnswer(
    const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const {
    if (!isReady_) {
        throw std::runtime_error("NeighborPIRServer not initialized");
    }
    if constexpr (sizeof(ElemType) == 4) {
        auto out = std::make_shared<MatrixT<ElemType>>(compressedDBRows_, queryMatrix->cols);
        if (queryMatrix->cols == 1) {
            matMulVecCompressed8_32(out->data.data(), compressedDB_.data(),
                                     queryMatrix->data.data(),
                                     compressedDBRows_, compressedDBCols_);
        } else {
            matMulCompressed8_32(out->data.data(), compressedDB_.data(),
                                  queryMatrix->data.data(),
                                  compressedDBRows_, compressedDBCols_, queryMatrix->cols);
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
void NeighborPIRClientT<ElemType>::init(
    const NeighborPIRParams& params,
    const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix,
    const std::shared_ptr<MatrixT<ElemType>>& hint
) {
    params_ = params;
    sharedMatrix_ = sharedMatrix;
    hint_ = hint;
    isInitialized_ = true;

    std::cout << "[NeighborPIRClient] Initialized. Hint: "
              << hint_->rows << " x " << hint_->cols << std::endl;
}

template<typename ElemType>
std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
NeighborPIRClientT<ElemType>::query(int subgroupColumn) {
    if (!isInitialized_) {
        throw std::runtime_error("NeighborPIRClient not initialized");
    }

    uint64_t N = params_.pirParams.N;
    uint64_t M = params_.pirParams.M;
    uint64_t logQ = params_.pirParams.Logq;

    auto secret = MatrixT<ElemType>::random(N, 1, logQ, 0);

    auto Hs = matrixMulVec(hint_, secret);

    auto queryVec = matrixMulVec(sharedMatrix_, secret);

    auto noise = MatrixT<ElemType>::gaussian(M, 1);
    queryVec->matrixAdd(*noise);

    uint64_t delta = params_.delta();
    uint64_t currentVal = queryVec->get(subgroupColumn, 0);
    queryVec->set(subgroupColumn, 0, currentVal + delta);

    NbrQueryMsgT<ElemType> msg;
    msg.queryVector = queryVec;
    msg.batchSize = 1;

    NbrQueryContextT<ElemType> ctx(Hs);

    return {msg, ctx};
}

template<typename ElemType>
std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
NeighborPIRClientT<ElemType>::query(
    int subgroupColumn,
    const std::shared_ptr<MatrixT<ElemType>>& precomputedAs,
    const std::shared_ptr<MatrixT<ElemType>>& precomputedHs
) {
    if (!isInitialized_) {
        throw std::runtime_error("NeighborPIRClient not initialized");
    }

    uint64_t M = params_.pirParams.M;

    auto queryVec = std::make_shared<MatrixT<ElemType>>(M, 1);
    std::memcpy(queryVec->data.data(), precomputedAs->data.data(),
                M * sizeof(ElemType));

    auto noise = MatrixT<ElemType>::gaussian(M, 1);
    queryVec->matrixAdd(*noise);

    uint64_t delta = params_.delta();
    uint64_t currentVal = queryVec->get(subgroupColumn, 0);
    queryVec->set(subgroupColumn, 0, currentVal + delta);

    NbrQueryMsgT<ElemType> msg;
    msg.queryVector = queryVec;
    msg.batchSize = 1;

    NbrQueryContextT<ElemType> ctx(precomputedHs);

    return {msg, ctx};
}

template<typename ElemType>
std::vector<uint64_t> NeighborPIRClientT<ElemType>::recoverNodeValues(
    const NbrAnswerMsgT<ElemType>& answer,
    const NbrQueryContextT<ElemType>& ctx,
    int localNodeIdx
) {
    if (!ctx.Hs) {
        throw std::invalid_argument("NbrQueryContext Hs cannot be null");
    }
    if (!answer.answer) {
        throw std::invalid_argument("Answer cannot be null");
    }

    int M0 = static_cast<int>(params_.config.maxNeighborsPerNode);
    int nParts = static_cast<int>(params_.config.numParts);
    int numValues = M0 * nParts;
    int rowStart = localNodeIdx * numValues;

    const auto* ansData = answer.answer->data.data();
    const auto* hsData = ctx.Hs->data.data();

    size_t ansSize = answer.answer->data.size();
    size_t hsSize = ctx.Hs->data.size();
    int maxRowIdx = rowStart + numValues - 1;

    if (maxRowIdx >= static_cast<int>(ansSize) || maxRowIdx >= static_cast<int>(hsSize)) {
        std::cerr << "[NeighborPIRClient::recoverNodeValues] ERROR: rowIdx out of bounds!\n"
                  << "  localNodeIdx=" << localNodeIdx << ", M0=" << M0 << ", numParts=" << nParts << "\n"
                  << "  rowStart=" << rowStart << ", maxRowIdx=" << maxRowIdx << "\n"
                  << "  ansSize=" << ansSize << ", hsSize=" << hsSize << "\n";
        return std::vector<uint64_t>(numValues, params_.pirParams.P - 1);
    }

    std::vector<uint64_t> results(numValues);

    for (int j = 0; j < numValues; ++j) {
        int rowIdx = rowStart + j;
        results[j] = recoverUnsignedValue<ElemType>(
            ansData[rowIdx], hsData[rowIdx], params_.pirParams);
    }

    return results;
}

template<typename ElemType>
NeighborPIRT<ElemType>::NeighborPIRT(const NeighborPIRConfig& config) : config_(config) {
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid NeighborPIRConfig");
    }
    params_.init(config);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> NeighborPIRT<ElemType>::generateSharedMatrix() {
    uint64_t rows = params_.pirParams.M;
    uint64_t cols = params_.pirParams.N;
    return MatrixT<ElemType>::random(rows, cols, params_.pirParams.Logq);
}

template<typename ElemType>
void NeighborPIRT<ElemType>::printBandwidth() const {
    uint64_t logq = params_.pirParams.Logq;
    uint64_t N = params_.pirParams.N;
    uint64_t L = params_.pirParams.L;
    uint64_t M = params_.pirParams.M;

    uint64_t elemBytes = sizeof(ElemType);

    uint64_t hintBytes = L * N * elemBytes;

    uint64_t queryBytes = M * elemBytes;

    uint64_t answerBytes = L * elemBytes;

    std::cout << "=== Neighbor PIR Bandwidth ===" << std::endl;
    std::cout << "Element size: " << elemBytes << " bytes (" << logq << "-bit logQ)" << std::endl;
    std::cout << "Offline download (Hint H): " << std::fixed << std::setprecision(2) << hintBytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Online upload (Query):     " << std::fixed << std::setprecision(2) << queryBytes / 1024.0 << " KB" << std::endl;
    std::cout << "Online download (Answer):  " << std::fixed << std::setprecision(2) << answerBytes / 1024.0 << " KB" << std::endl;
    std::cout << "==============================" << std::endl;
}

template struct NbrQueryContextT<Elem32>;
template struct NbrQueryContextT<Elem64>;
template struct NbrQueryMsgT<Elem32>;
template struct NbrQueryMsgT<Elem64>;
template struct NbrAnswerMsgT<Elem32>;
template struct NbrAnswerMsgT<Elem64>;
template class NeighborDatabaseT<Elem32>;
template class NeighborDatabaseT<Elem64>;
template class NeighborPIRServerT<Elem32>;
template class NeighborPIRServerT<Elem64>;
template class NeighborPIRClientT<Elem32>;
template class NeighborPIRClientT<Elem64>;
template class NeighborPIRT<Elem32>;
template class NeighborPIRT<Elem64>;

}

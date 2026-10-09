#include "simple_pir.h"
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace simplepir {

std::shared_ptr<Matrix> SimplePIRServer::setup(
    const std::shared_ptr<Matrix>& database,
    const Params& params,
    const std::shared_ptr<Matrix>& sharedMatrix
) {
    if (!database || !sharedMatrix) {
        throw std::invalid_argument("Database and sharedMatrix cannot be null");
    }

    database_ = database;
    params_ = params;
    sharedMatrix_ = sharedMatrix;

    hint_ = matrixMul(database_, sharedMatrix);

    isReady_ = true;
    return hint_;
}

std::shared_ptr<Matrix> SimplePIRServer::answer(
    const std::shared_ptr<Matrix>& queryVector
) const {
    if (!isReady_) {
        throw std::runtime_error("Server not initialized");
    }
    if (!queryVector) {
        throw std::invalid_argument("Query vector cannot be null");
    }

    return matrixMulVec(database_, queryVector);
}

void SimplePIRClient::init(
    const Params& params,
    const std::shared_ptr<Matrix>& sharedMatrix,
    const std::shared_ptr<Matrix>& hint
) {
    if (!sharedMatrix || !hint) {
        throw std::invalid_argument("sharedMatrix and hint cannot be null");
    }

    params_ = params;
    sharedMatrix_ = sharedMatrix;
    hint_ = hint;

    isInitialized_ = true;
}

std::pair<std::shared_ptr<Matrix>, SimpleQueryContext> SimplePIRClient::query(uint64_t targetCol) {
    if (!isInitialized_) {
        throw std::runtime_error("Client not initialized");
    }
    if (targetCol >= params_.M) {
        throw std::out_of_range("Target column out of range");
    }

    auto secret = Matrix::random(params_.N, 1, params_.Logq, 0);

    auto Hs = matrixMulVec(hint_, secret);

    auto queryVec = matrixMulVec(sharedMatrix_, secret);

    auto noise = Matrix::gaussian(params_.M, 1);
    queryVec->matrixAdd(*noise);

    uint64_t delta = params_.delta();
    uint64_t currentVal = queryVec->get(targetCol, 0);
    queryVec->set(targetCol, 0, currentVal + delta);

    SimpleQueryContext ctx(Hs);

    return {queryVec, ctx};
}

std::vector<uint64_t> SimplePIRClient::recover(
    const std::shared_ptr<Matrix>& answer,
    const SimpleQueryContext& ctx
) {
    if (!ctx.Hs) {
        throw std::invalid_argument("QueryContext Hs cannot be null");
    }
    if (!answer) {
        throw std::invalid_argument("Answer cannot be null");
    }

    const uint64_t L = params_.L;
    const uint64_t P = params_.P;
    const uint64_t delta = params_.delta();
    const uint64_t halfDelta = delta / 2;
    const uint64_t q_mod = 1ULL << params_.Logq;

    std::vector<uint64_t> result(L);

    const Elem* ansData = answer->data.data();
    const Elem* hsData = ctx.Hs->data.data();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (uint64_t i = 0; i < L; ++i) {
        uint64_t ansVal = ansData[i];
        uint64_t hsVal = hsData[i];

        uint64_t diff;
        if (ansVal >= hsVal) {
            diff = ansVal - hsVal;
        } else {
            diff = q_mod - (hsVal - ansVal);
        }

        int64_t signedDiff;
        if (diff > q_mod / 2) {
            signedDiff = static_cast<int64_t>(diff) - static_cast<int64_t>(q_mod);
        } else {
            signedDiff = static_cast<int64_t>(diff);
        }

        int64_t rounded = (signedDiff + static_cast<int64_t>(halfDelta)) / static_cast<int64_t>(delta);

        if (rounded < 0) rounded = 0;
        if (rounded >= static_cast<int64_t>(P)) rounded = P - 1;

        result[i] = static_cast<uint64_t>(rounded);
    }

    return result;
}

}

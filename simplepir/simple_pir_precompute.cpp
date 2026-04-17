/**
 * @file simple_pir_precompute.cpp
 * @brief Implementation of SimplePIR with precomputation
 *
 * Optimization:
 * - Uses matrix.h wrappers which call OpenMP + AVX2 optimized pir_math functions
 * - Parallel recovery with OpenMP
 */

#include "simple_pir_precompute.h"
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace simplepir {

// ============================================================================
// SimplePIRServer Implementation
// ============================================================================

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

    // Compute hint: H = DB × A
    // Uses matrix.h wrapper → pir_math OpenMP + AVX2
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

    // ans = DB × query
    // Uses matrix.h wrapper → pir_math OpenMP + AVX2
    return matrixMulVec(database_, queryVector);
}

// ============================================================================
// SimplePIRClient Implementation
// ============================================================================

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
    isPrecomputed_ = false;
}

void SimplePIRClient::precompute() {
    if (!isInitialized_) {
        throw std::runtime_error("Client not initialized");
    }

    // Generate new secret s (N × 1)
    secret_ = Matrix::random(params_.N, 1, params_.Logq, 0);

    // Precompute Hs = H × s (L × 1)
    // Uses matrix.h wrapper → pir_math OpenMP + AVX2
    Hs_ = matrixMulVec(hint_, secret_);

    // Precompute baseQuery = A × s (M × 1)
    // Uses matrix.h wrapper → pir_math OpenMP + AVX2
    baseQuery_ = matrixMulVec(sharedMatrix_, secret_);

    isPrecomputed_ = true;
}

std::shared_ptr<Matrix> SimplePIRClient::query(uint64_t targetCol) {
    if (!isPrecomputed_) {
        throw std::runtime_error("Must call precompute() before query()");
    }
    if (targetCol >= params_.M) {
        throw std::out_of_range("Target column out of range");
    }

    // Copy baseQuery
    auto queryVec = std::make_shared<Matrix>(*baseQuery_);

    // Add Gaussian noise
    auto noise = Matrix::gaussian(params_.M, 1);
    queryVec->matrixAdd(*noise);

    // Add delta at target column
    uint64_t delta = params_.delta();
    uint64_t currentVal = queryVec->get(targetCol, 0);
    queryVec->set(targetCol, 0, currentVal + delta);

    return queryVec;
}

std::vector<uint64_t> SimplePIRClient::recover(
    const std::shared_ptr<Matrix>& answer
) {
    if (!isPrecomputed_) {
        throw std::runtime_error("Must call precompute() before recover()");
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
    const Elem* hsData = Hs_->data.data();

    // OpenMP parallelized recovery
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (uint64_t i = 0; i < L; ++i) {
        uint64_t ansVal = ansData[i];
        uint64_t hsVal = hsData[i];

        // Compute difference (modular arithmetic)
        uint64_t diff;
        if (ansVal >= hsVal) {
            diff = ansVal - hsVal;
        } else {
            diff = q_mod - (hsVal - ansVal);
        }

        // Use signed interpretation for values near 0
        // If diff > Q/2, it wrapped around (original value was near 0)
        int64_t signedDiff;
        if (diff > q_mod / 2) {
            signedDiff = static_cast<int64_t>(diff) - static_cast<int64_t>(q_mod);
        } else {
            signedDiff = static_cast<int64_t>(diff);
        }

        // Round to recover original value
        int64_t rounded = (signedDiff + static_cast<int64_t>(halfDelta)) / static_cast<int64_t>(delta);

        // Clamp to valid range [0, P-1]
        if (rounded < 0) rounded = 0;
        if (rounded >= static_cast<int64_t>(P)) rounded = P - 1;

        result[i] = static_cast<uint64_t>(rounded);
    }

    return result;
}

} // namespace simplepir

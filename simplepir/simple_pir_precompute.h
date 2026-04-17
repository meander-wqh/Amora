/**
 * @file simple_pir_precompute.h
 * @brief SimplePIR with precomputation optimization
 *
 * This provides a stateful wrapper around SimplePIR that supports:
 * - Precomputation of Hs = H × s and baseQuery = A × s
 * - Efficient multiple queries using precomputed values
 * - Independent parameters from EmbeddingPIR
 */

#ifndef SIMPLE_PIR_PRECOMPUTE_H
#define SIMPLE_PIR_PRECOMPUTE_H

#include "pir_types.h"
#include "matrix.h"
#include <memory>
#include <vector>

namespace simplepir {

/**
 * @brief SimplePIR Server
 *
 * Handles database setup and query answering for general PIR queries.
 */
class SimplePIRServer {
public:
    SimplePIRServer() = default;

    /**
     * @brief Setup server with database
     * @param database L × M matrix
     * @param params PIR parameters
     * @param sharedMatrix M × N shared random matrix A
     * @return Hint matrix H = DB × A (L × N)
     */
    std::shared_ptr<Matrix> setup(
        const std::shared_ptr<Matrix>& database,
        const Params& params,
        const std::shared_ptr<Matrix>& sharedMatrix
    );

    /**
     * @brief Answer a query
     * @param queryVector Query vector (M × 1)
     * @return Answer vector = DB × query (L × 1)
     */
    std::shared_ptr<Matrix> answer(const std::shared_ptr<Matrix>& queryVector) const;

    /**
     * @brief Get hint matrix
     */
    std::shared_ptr<Matrix> getHint() const { return hint_; }

    /**
     * @brief Get shared matrix A
     */
    std::shared_ptr<Matrix> getSharedMatrix() const { return sharedMatrix_; }

    /**
     * @brief Get database dimensions
     */
    uint64_t getRows() const { return database_ ? database_->rows : 0; }
    uint64_t getCols() const { return database_ ? database_->cols : 0; }

    /**
     * @brief Check if server is ready
     */
    bool isReady() const { return isReady_; }

private:
    std::shared_ptr<Matrix> database_;      // L × M
    std::shared_ptr<Matrix> sharedMatrix_;  // A (M × N)
    std::shared_ptr<Matrix> hint_;          // H = DB × A (L × N)
    Params params_;
    bool isReady_ = false;
};

/**
 * @brief SimplePIR Client with precomputation optimization
 *
 * Supports efficient multiple queries by precomputing:
 * - Hs = H × s (once per ANN query)
 * - baseQuery = A × s (once per ANN query)
 */
class SimplePIRClient {
public:
    SimplePIRClient() = default;

    /**
     * @brief Initialize client
     * @param params PIR parameters (must set N, M, L, Logq, P)
     * @param sharedMatrix Shared random matrix A (M × N)
     * @param hint Hint matrix H from server (L × N)
     */
    void init(
        const Params& params,
        const std::shared_ptr<Matrix>& sharedMatrix,
        const std::shared_ptr<Matrix>& hint
    );

    /**
     * @brief Precompute Hs and baseQuery for a new ANN query
     *
     * Call this once at the start of each ANN query.
     * Generates new secret s and computes:
     * - Hs = H × s
     * - baseQuery = A × s
     */
    void precompute();

    /**
     * @brief Generate query for target column
     * @param targetCol Target column index [0, M-1]
     * @return Query vector (M × 1)
     *
     * query = baseQuery + noise + delta * e_{targetCol}
     */
    std::shared_ptr<Matrix> query(uint64_t targetCol);

    /**
     * @brief Recover values from answer using precomputed Hs
     * @param answer Answer vector from server (L × 1)
     * @return Recovered values (L elements, each in [0, P-1])
     *
     * For each row i: recovered[i] = round((answer[i] - Hs[i]) / delta)
     */
    std::vector<uint64_t> recover(const std::shared_ptr<Matrix>& answer);

    /**
     * @brief Check if client is initialized
     */
    bool isInitialized() const { return isInitialized_; }

    /**
     * @brief Check if precomputation is done
     */
    bool isPrecomputed() const { return isPrecomputed_; }

    /**
     * @brief Get parameters
     */
    const Params& getParams() const { return params_; }

    /**
     * @brief Get hint matrix (for external access if needed)
     */
    std::shared_ptr<Matrix> getHint() const { return hint_; }

    /**
     * @brief Get shared matrix (for external access if needed)
     */
    std::shared_ptr<Matrix> getSharedMatrix() const { return sharedMatrix_; }

    /**
     * @brief Get precomputed Hs (for external optimization)
     */
    std::shared_ptr<Matrix> getHs() const { return Hs_; }

    /**
     * @brief Get precomputed baseQuery (for external optimization)
     */
    std::shared_ptr<Matrix> getBaseQuery() const { return baseQuery_; }

private:
    Params params_;
    std::shared_ptr<Matrix> sharedMatrix_;  // A (M × N)
    std::shared_ptr<Matrix> hint_;          // H (L × N)
    std::shared_ptr<Matrix> secret_;        // s (N × 1)
    std::shared_ptr<Matrix> Hs_;            // Precomputed: H × s (L × 1)
    std::shared_ptr<Matrix> baseQuery_;     // Precomputed: A × s (M × 1)
    bool isInitialized_ = false;
    bool isPrecomputed_ = false;
};

} // namespace simplepir

#endif // SIMPLE_PIR_PRECOMPUTE_H

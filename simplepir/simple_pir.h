/**
 * @file simple_pir.h
 * @brief SimplePIR（安全版本）
 *
 * 每次query都生成新的secret s，避免复用s导致的隐私泄露。
 *
 * 安全问题说明：
 * 如果复用同一个s，服务器可以计算 qu_1 - qu_2 = (e_1 - e_2) + Δ·(u_i - u_j)
 * 由于 Δ = ⌊q/p⌋ 很大而噪声很小，服务器可以直接观察出查询的目标索引差异。
 * 因此每次查询必须使用新的s。
 */

#ifndef SIMPLE_PIR_H
#define SIMPLE_PIR_H

#include "pir_types.h"
#include "matrix.h"
#include <memory>
#include <vector>
#include <utility>

namespace simplepir {

/**
 * @brief 查询上下文，包含恢复所需的状态（32位版本）
 *
 * 每次PIR查询都使用新的secret s，对应的Hs保存在此结构中用于recover。
 */
struct SimpleQueryContext {
    std::shared_ptr<Matrix> Hs;  // H × s，用于recover

    SimpleQueryContext() = default;
    explicit SimpleQueryContext(std::shared_ptr<Matrix> hs) : Hs(std::move(hs)) {}
};

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
 * @brief SimplePIR Client（安全版本）
 *
 * 每次query都生成新的secret s，避免复用s导致的隐私泄露。
 *
 * Usage:
 *   client.init(params, sharedMatrix, hint);
 *   auto [query, ctx] = client.query(targetCol);
 *   auto answer = server.answer(query);
 *   auto result = client.recover(answer, ctx);
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
     * @brief Generate query for target column（安全版本）
     * @param targetCol Target column index [0, M-1]
     * @return pair<Query vector, QueryContext>
     *
     * 每次调用都生成新的secret s，保证PIR隐私性。
     * query = A×s + noise + delta * e_{targetCol}
     */
    std::pair<std::shared_ptr<Matrix>, SimpleQueryContext> query(uint64_t targetCol);

    /**
     * @brief Recover values from answer
     * @param answer Answer vector from server (L × 1)
     * @param ctx Query context containing Hs for this query
     * @return Recovered values (L elements, each in [0, P-1])
     *
     * For each row i: recovered[i] = round((answer[i] - Hs[i]) / delta)
     */
    std::vector<uint64_t> recover(const std::shared_ptr<Matrix>& answer, const SimpleQueryContext& ctx);

    /**
     * @brief Check if client is initialized
     */
    bool isInitialized() const { return isInitialized_; }

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

private:
    Params params_;
    std::shared_ptr<Matrix> sharedMatrix_;  // A (M × N)
    std::shared_ptr<Matrix> hint_;          // H (L × N)
    bool isInitialized_ = false;
};

} // namespace simplepir

#endif // SIMPLE_PIR_H

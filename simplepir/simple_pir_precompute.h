#ifndef SIMPLE_PIR_PRECOMPUTE_H
#define SIMPLE_PIR_PRECOMPUTE_H

#include "pir_types.h"
#include "matrix.h"
#include <memory>
#include <vector>

namespace simplepir {

class SimplePIRServer {
public:
    SimplePIRServer() = default;

    std::shared_ptr<Matrix> setup(
        const std::shared_ptr<Matrix>& database,
        const Params& params,
        const std::shared_ptr<Matrix>& sharedMatrix
    );

    std::shared_ptr<Matrix> answer(const std::shared_ptr<Matrix>& queryVector) const;

    std::shared_ptr<Matrix> getHint() const { return hint_; }

    std::shared_ptr<Matrix> getSharedMatrix() const { return sharedMatrix_; }

    uint64_t getRows() const { return database_ ? database_->rows : 0; }
    uint64_t getCols() const { return database_ ? database_->cols : 0; }

    bool isReady() const { return isReady_; }

private:
    std::shared_ptr<Matrix> database_;
    std::shared_ptr<Matrix> sharedMatrix_;
    std::shared_ptr<Matrix> hint_;
    Params params_;
    bool isReady_ = false;
};

class SimplePIRClient {
public:
    SimplePIRClient() = default;

    void init(
        const Params& params,
        const std::shared_ptr<Matrix>& sharedMatrix,
        const std::shared_ptr<Matrix>& hint
    );

    void precompute();

    std::shared_ptr<Matrix> query(uint64_t targetCol);

    std::vector<uint64_t> recover(const std::shared_ptr<Matrix>& answer);

    bool isInitialized() const { return isInitialized_; }

    bool isPrecomputed() const { return isPrecomputed_; }

    const Params& getParams() const { return params_; }

    std::shared_ptr<Matrix> getHint() const { return hint_; }

    std::shared_ptr<Matrix> getSharedMatrix() const { return sharedMatrix_; }

    std::shared_ptr<Matrix> getHs() const { return Hs_; }

    std::shared_ptr<Matrix> getBaseQuery() const { return baseQuery_; }

private:
    Params params_;
    std::shared_ptr<Matrix> sharedMatrix_;
    std::shared_ptr<Matrix> hint_;
    std::shared_ptr<Matrix> secret_;
    std::shared_ptr<Matrix> Hs_;
    std::shared_ptr<Matrix> baseQuery_;
    bool isInitialized_ = false;
    bool isPrecomputed_ = false;
};

}

#endif

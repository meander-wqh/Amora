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

struct NeighborPIRConfig {
    uint64_t totalSubgroups = 0;
    uint64_t maxSubgroupSize = 0;
    uint64_t maxNeighborsPerNode = 0;
    uint64_t numParts = 1;

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
    Params pirParams;

    uint32_t invalidNeighbor() const { return static_cast<uint32_t>(pirParams.P - 1); }

    uint64_t delta() const { return pirParams.delta(); }

    void init(const NeighborPIRConfig& cfg,
              uint64_t logQ = 64, uint64_t lweN = 1024, double sigma = 6.4,
              uint64_t P = (1ULL << 22));
};

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

template<typename ElemType>
class NeighborDatabaseT {
public:
    NeighborDatabaseT() = default;
    explicit NeighborDatabaseT(const NeighborPIRConfig& config);

    void setEncodedNeighbor(int subgroupColumn, int localNodeIdx,
                            int neighborSlot, uint64_t encodedValue);

    void setElement(int row, int col, ElemType value);

    std::shared_ptr<MatrixT<ElemType>> getMatrix() const { return data_; }
    const NeighborPIRConfig& getConfig() const { return config_; }
    bool isReady() const { return isInitialized_; }
    void printInfo() const;

    std::vector<uint8_t>& compressedDataRef() { return compressedData_; }
    uint64_t getDBRows() const { return dbRows_; }
    uint64_t getDBCols() const { return dbCols_; }
    bool hasCompressedData() const { return !compressedData_.empty(); }

private:
    NeighborPIRConfig config_;
    std::shared_ptr<MatrixT<ElemType>> data_;
    std::vector<uint8_t> compressedData_;
    uint64_t dbRows_ = 0, dbCols_ = 0;
    bool isInitialized_ = false;
};
using NeighborDatabase = NeighborDatabaseT<Elem64>;

template<typename ElemType>
class NeighborPIRServerT {
public:
    NeighborPIRServerT() = default;

    std::shared_ptr<MatrixT<ElemType>> setup(
        NeighborDatabaseT<ElemType>& db,
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix
    );

    void setupWithCache(
        NeighborDatabaseT<ElemType>& db,
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& cachedHint
    );

    NbrAnswerMsgT<ElemType> answer(const NbrQueryMsgT<ElemType>& query) const;

    std::shared_ptr<MatrixT<ElemType>> batchAnswer(
        const std::shared_ptr<MatrixT<ElemType>>& queryMatrix) const;

    bool isReady() const { return isReady_; }

private:
    std::shared_ptr<MatrixT<ElemType>> database_;
    std::shared_ptr<MatrixT<ElemType>> hint_;
    NeighborPIRParams params_;
    bool isReady_ = false;

    std::vector<uint8_t> compressedDB_;
    uint64_t compressedDBRows_ = 0;
    uint64_t compressedDBCols_ = 0;
};
using NeighborPIRServer = NeighborPIRServerT<Elem64>;

template<typename ElemType>
class NeighborPIRClientT {
public:
    NeighborPIRClientT() = default;

    void init(
        const NeighborPIRParams& params,
        const std::shared_ptr<MatrixT<ElemType>>& sharedMatrix,
        const std::shared_ptr<MatrixT<ElemType>>& hint
    );

    std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
    query(int subgroupColumn);

    std::pair<NbrQueryMsgT<ElemType>, NbrQueryContextT<ElemType>>
    query(int subgroupColumn,
          const std::shared_ptr<MatrixT<ElemType>>& precomputedAs,
          const std::shared_ptr<MatrixT<ElemType>>& precomputedHs);

    std::vector<uint64_t> recoverNodeValues(
        const NbrAnswerMsgT<ElemType>& answer,
        const NbrQueryContextT<ElemType>& ctx,
        int localNodeIdx
    );

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

}

#endif

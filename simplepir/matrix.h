#ifndef PIR_MATRIX_H
#define PIR_MATRIX_H
#include "pir_types.h"
#include "pir_math.h"
#include "random.h"
#include <iostream>
#include <algorithm>
namespace simplepir {

/**
 * @brief Templated Matrix class for PIR operations
 * @tparam ElemType Element type (Elem32 for 32-bit, Elem64 for 64-bit)
 */
template<typename ElemType>
class MatrixT {
public:
    uint64_t rows = 0, cols = 0;
    std::vector<ElemType> data;

    MatrixT() = default;
    MatrixT(uint64_t r, uint64_t c) : rows(r), cols(c), data(r * c, 0) {}
    MatrixT(uint64_t r, uint64_t c, bool alloc) : rows(r), cols(c) { if (alloc) data.resize(r * c, 0); }

    static std::shared_ptr<MatrixT> zeros(uint64_t r, uint64_t c);
    static std::shared_ptr<MatrixT> random(uint64_t r, uint64_t c, uint64_t logMod, uint64_t mod = 0);
    static std::shared_ptr<MatrixT> gaussian(uint64_t r, uint64_t c);

    uint64_t size() const { return rows * cols; }
    uint64_t sizeBytes() const { return rows * cols * sizeof(ElemType); }
    static constexpr uint64_t elemSize() { return sizeof(ElemType); }

    uint64_t get(uint64_t i, uint64_t j) const;
    void set(uint64_t i, uint64_t j, uint64_t val);
    void addAt(uint64_t val, uint64_t i, uint64_t j);
    void appendZeros(uint64_t n);
    void dropLastRows(uint64_t n);

    void matrixAdd(const MatrixT& other);
    void matrixSub(const MatrixT& other);
    void add(uint64_t val);
    void sub(uint64_t val);
    void reduceMod(uint64_t p);
    void round(const Params& p);
    void transpose();
    void concat(const MatrixT& other);
    void concat(const std::shared_ptr<MatrixT>& other);

    std::shared_ptr<MatrixT> selectColumn(uint64_t i) const;
    std::shared_ptr<MatrixT> selectRows(uint64_t offset, uint64_t numRows) const;
    std::shared_ptr<MatrixT> rowsDeepCopy(uint64_t offset, uint64_t numRows) const;

    void expand(uint64_t mod, uint64_t delta);
    void contract(uint64_t mod, uint64_t delta);
    void squish(uint64_t basis, uint64_t delta);
    void unsquish(uint64_t basis, uint64_t delta, uint64_t originalCols);
    void concatCols(uint64_t n);

    void dim() const;
    void print() const;
};

// Matrix multiplication functions (templated)
template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMul(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMul(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVec(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVec(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVecPacked(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b, uint64_t basis, uint64_t compression);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVecPacked(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b, uint64_t basis, uint64_t compression);

// Squish/Unsquish helper functions
template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> squishMatrix(const std::shared_ptr<MatrixT<ElemType>>& m, uint64_t basis, uint64_t delta);

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> unsquishMatrix(const std::shared_ptr<MatrixT<ElemType>>& m, uint64_t basis, uint64_t delta);

// Note: Matrix and Matrix64 type aliases are defined in pir_types.h

// Explicit template instantiation declarations
extern template class MatrixT<Elem32>;
extern template class MatrixT<Elem64>;

} // namespace simplepir
#endif // PIR_MATRIX_H

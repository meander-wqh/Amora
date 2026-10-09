#include "matrix.h"
#include <stdexcept>
namespace simplepir {

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::zeros(uint64_t r, uint64_t c) {
    auto m = std::make_shared<MatrixT<ElemType>>(r, c);
    std::fill(m->data.begin(), m->data.end(), 0);
    return m;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::random(uint64_t r, uint64_t c, uint64_t logMod, uint64_t mod) {
    auto m = std::make_shared<MatrixT<ElemType>>(r, c);
    uint64_t modulus = mod == 0 ? (1ULL << logMod) : mod;
    for (auto& v : m->data) v = static_cast<ElemType>(randInt(modulus));
    return m;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::gaussian(uint64_t r, uint64_t c) {
    auto m = std::make_shared<MatrixT<ElemType>>(r, c);
    for (auto& v : m->data) v = static_cast<ElemType>(gaussSample());
    return m;
}

template<typename ElemType>
uint64_t MatrixT<ElemType>::get(uint64_t i, uint64_t j) const {
    if (i >= rows || j >= cols) throw std::out_of_range("Index out of range");
    return static_cast<uint64_t>(data[i * cols + j]);
}

template<typename ElemType>
void MatrixT<ElemType>::set(uint64_t i, uint64_t j, uint64_t val) {
    if (i >= rows || j >= cols) throw std::out_of_range("Index out of range");
    data[i * cols + j] = static_cast<ElemType>(val);
}

template<typename ElemType>
void MatrixT<ElemType>::addAt(uint64_t val, uint64_t i, uint64_t j) { set(i, j, get(i, j) + val); }

template<typename ElemType>
void MatrixT<ElemType>::appendZeros(uint64_t n) { concat(*MatrixT<ElemType>::zeros(n, 1)); }

template<typename ElemType>
void MatrixT<ElemType>::dropLastRows(uint64_t n) { rows -= n; data.resize(rows * cols); }

template<typename ElemType>
void MatrixT<ElemType>::matrixAdd(const MatrixT<ElemType>& other) {
    if (cols != other.cols || rows != other.rows) throw std::invalid_argument("Dimension mismatch");
    for (size_t i = 0; i < data.size(); i++) data[i] += other.data[i];
}

template<typename ElemType>
void MatrixT<ElemType>::matrixSub(const MatrixT<ElemType>& other) {
    if (cols != other.cols || rows != other.rows) throw std::invalid_argument("Dimension mismatch");
    for (size_t i = 0; i < data.size(); i++) data[i] -= other.data[i];
}

template<typename ElemType>
void MatrixT<ElemType>::add(uint64_t val) { ElemType v = static_cast<ElemType>(val); for (auto& d : data) d += v; }

template<typename ElemType>
void MatrixT<ElemType>::sub(uint64_t val) { ElemType v = static_cast<ElemType>(val); for (auto& d : data) d -= v; }

template<typename ElemType>
void MatrixT<ElemType>::reduceMod(uint64_t p) { ElemType mod = static_cast<ElemType>(p); for (auto& d : data) d = d % mod; }

template<typename ElemType>
void MatrixT<ElemType>::round(const Params& p) { for (auto& d : data) d = static_cast<ElemType>(p.round(d)); }

template<typename ElemType>
void MatrixT<ElemType>::transpose() {
    if (cols == 1) { cols = rows; rows = 1; return; }
    if (rows == 1) { rows = cols; cols = 1; return; }
    std::vector<ElemType> newData(rows * cols);
    matrixTransposeT<ElemType>(newData.data(), data.data(), rows, cols);
    std::swap(rows, cols);
    data = std::move(newData);
}

template<typename ElemType>
void MatrixT<ElemType>::concat(const MatrixT<ElemType>& other) {
    if (cols == 0 && rows == 0) { cols = other.cols; rows = other.rows; data = other.data; return; }
    if (cols != other.cols) throw std::invalid_argument("Dimension mismatch");
    rows += other.rows;
    data.insert(data.end(), other.data.begin(), other.data.end());
}

template<typename ElemType>
void MatrixT<ElemType>::concat(const std::shared_ptr<MatrixT<ElemType>>& other) { if (other) concat(*other); }

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::selectColumn(uint64_t i) const {
    if (cols == 1) { auto r = std::make_shared<MatrixT<ElemType>>(rows, 1, false); r->data = data; return r; }
    auto col = std::make_shared<MatrixT<ElemType>>(rows, 1);
    for (uint64_t j = 0; j < rows; j++) col->data[j] = data[j * cols + i];
    return col;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::selectRows(uint64_t offset, uint64_t numRows) const {
    if (offset == 0 && numRows == rows) { auto r = std::make_shared<MatrixT<ElemType>>(rows, cols, false); r->data = data; return r; }
    uint64_t actualNumRows = std::min(numRows, rows - offset);
    auto m2 = std::make_shared<MatrixT<ElemType>>(actualNumRows, cols, false);
    m2->data.assign(data.begin() + offset * cols, data.begin() + (offset + actualNumRows) * cols);
    return m2;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> MatrixT<ElemType>::rowsDeepCopy(uint64_t offset, uint64_t numRows) const {
    auto m2 = std::make_shared<MatrixT<ElemType>>(numRows, cols);
    std::copy(data.begin() + offset * cols, data.begin() + (offset + numRows) * cols, m2->data.begin());
    return m2;
}

template<typename ElemType>
void MatrixT<ElemType>::squish(uint64_t basis, uint64_t delta) {
    auto n = MatrixT<ElemType>::zeros(rows, (cols + delta - 1) / delta);
    for (uint64_t i = 0; i < n->rows; i++)
        for (uint64_t j = 0; j < n->cols; j++)
            for (uint64_t k = 0; k < delta && delta * j + k < cols; k++)
                n->data[i * n->cols + j] += static_cast<ElemType>(get(i, delta * j + k) << (k * basis));
    cols = n->cols; rows = n->rows; data = std::move(n->data);
}

template<typename ElemType>
void MatrixT<ElemType>::unsquish(uint64_t basis, uint64_t delta, uint64_t originalCols) {
    auto n = MatrixT<ElemType>::zeros(rows, originalCols);
    uint64_t mask = (1ULL << basis) - 1;
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            for (uint64_t k = 0; k < delta && j * delta + k < originalCols; k++)
                n->data[i * n->cols + j * delta + k] = static_cast<ElemType>((get(i, j) >> (k * basis)) & mask);
    cols = n->cols; rows = n->rows; data = std::move(n->data);
}

template<typename ElemType>
void MatrixT<ElemType>::dim() const { std::cout << "Dims: " << rows << "-by-" << cols << std::endl; }

template<typename ElemType>
void MatrixT<ElemType>::print() const {
    std::cout << rows << "-by-" << cols << " matrix:" << std::endl;
    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j < cols; j++) std::cout << data[i * cols + j] << " ";
        std::cout << std::endl;
    }
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMul(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b) {
    if (b.cols == 1) return matrixMulVec(a, b);
    auto out = MatrixT<ElemType>::zeros(a.rows, b.cols);
    matMulT<ElemType>(out->data.data(), a.data.data(), b.data.data(), a.rows, a.cols, b.cols);
    return out;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMul(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b) {
    return matrixMul(*a, *b);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVec(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b) {
    auto out = std::make_shared<MatrixT<ElemType>>(a.rows, 1);
    matMulVecT<ElemType>(out->data.data(), a.data.data(), b.data.data(), a.rows, a.cols);
    return out;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVec(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b) {
    return matrixMulVec(*a, *b);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVecPacked(const MatrixT<ElemType>& a, const MatrixT<ElemType>& b, uint64_t basis, uint64_t compression) {
    auto out = std::make_shared<MatrixT<ElemType>>(a.rows + 8, 1);
    if constexpr (sizeof(ElemType) == 4) {
        matMulVecPacked32(out->data.data(), a.data.data(), b.data.data(), a.rows, a.cols);
    } else {
        matMulVecT<ElemType>(out->data.data(), a.data.data(), b.data.data(), a.rows, a.cols);
    }
    out->dropLastRows(8);
    return out;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> matrixMulVecPacked(const std::shared_ptr<MatrixT<ElemType>>& a, const std::shared_ptr<MatrixT<ElemType>>& b, uint64_t basis, uint64_t compression) {
    return matrixMulVecPacked(*a, *b, basis, compression);
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> squishMatrix(const std::shared_ptr<MatrixT<ElemType>>& m, uint64_t basis, uint64_t delta) {
    auto result = std::make_shared<MatrixT<ElemType>>(*m);
    result->squish(basis, delta);
    return result;
}

template<typename ElemType>
std::shared_ptr<MatrixT<ElemType>> unsquishMatrix(const std::shared_ptr<MatrixT<ElemType>>& m, uint64_t basis, uint64_t delta) {
    uint64_t originalCols = m->cols * delta;
    auto result = std::make_shared<MatrixT<ElemType>>(*m);
    result->unsquish(basis, delta, originalCols);
    return result;
}

template class MatrixT<Elem32>;
template class MatrixT<Elem64>;

template std::shared_ptr<MatrixT<Elem32>> matrixMul(const MatrixT<Elem32>&, const MatrixT<Elem32>&);
template std::shared_ptr<MatrixT<Elem64>> matrixMul(const MatrixT<Elem64>&, const MatrixT<Elem64>&);

template std::shared_ptr<MatrixT<Elem32>> matrixMul(const std::shared_ptr<MatrixT<Elem32>>&, const std::shared_ptr<MatrixT<Elem32>>&);
template std::shared_ptr<MatrixT<Elem64>> matrixMul(const std::shared_ptr<MatrixT<Elem64>>&, const std::shared_ptr<MatrixT<Elem64>>&);

template std::shared_ptr<MatrixT<Elem32>> matrixMulVec(const MatrixT<Elem32>&, const MatrixT<Elem32>&);
template std::shared_ptr<MatrixT<Elem64>> matrixMulVec(const MatrixT<Elem64>&, const MatrixT<Elem64>&);

template std::shared_ptr<MatrixT<Elem32>> matrixMulVec(const std::shared_ptr<MatrixT<Elem32>>&, const std::shared_ptr<MatrixT<Elem32>>&);
template std::shared_ptr<MatrixT<Elem64>> matrixMulVec(const std::shared_ptr<MatrixT<Elem64>>&, const std::shared_ptr<MatrixT<Elem64>>&);

template std::shared_ptr<MatrixT<Elem32>> matrixMulVecPacked(const MatrixT<Elem32>&, const MatrixT<Elem32>&, uint64_t, uint64_t);
template std::shared_ptr<MatrixT<Elem64>> matrixMulVecPacked(const MatrixT<Elem64>&, const MatrixT<Elem64>&, uint64_t, uint64_t);

template std::shared_ptr<MatrixT<Elem32>> matrixMulVecPacked(const std::shared_ptr<MatrixT<Elem32>>&, const std::shared_ptr<MatrixT<Elem32>>&, uint64_t, uint64_t);
template std::shared_ptr<MatrixT<Elem64>> matrixMulVecPacked(const std::shared_ptr<MatrixT<Elem64>>&, const std::shared_ptr<MatrixT<Elem64>>&, uint64_t, uint64_t);

template std::shared_ptr<MatrixT<Elem32>> squishMatrix(const std::shared_ptr<MatrixT<Elem32>>&, uint64_t, uint64_t);
template std::shared_ptr<MatrixT<Elem64>> squishMatrix(const std::shared_ptr<MatrixT<Elem64>>&, uint64_t, uint64_t);

template std::shared_ptr<MatrixT<Elem32>> unsquishMatrix(const std::shared_ptr<MatrixT<Elem32>>&, uint64_t, uint64_t);
template std::shared_ptr<MatrixT<Elem64>> unsquishMatrix(const std::shared_ptr<MatrixT<Elem64>>&, uint64_t, uint64_t);

}

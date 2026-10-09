#ifndef PIR_MATH_H
#define PIR_MATH_H
#include "pir_types.h"
namespace simplepir {

void matrixTranspose32(Elem32* out, const Elem32* in, size_t rows, size_t cols);
void matMul32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bCols);
void matMulTransposedPacked32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bRows, size_t bCols);
void matMulVec32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols);
void matMulVecPacked32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols);

void matMulVecCompressed8_32(Elem32* out, const uint8_t* db, const Elem32* query,
                              size_t dbRows, size_t dbCols, bool signedData = false);

void matMulCompressed8_32(Elem32* out, const uint8_t* db, const Elem32* b,
                           size_t dbRows, size_t dbCols, size_t bCols, bool signedData = false);

void matrixTranspose64(Elem64* out, const Elem64* in, size_t rows, size_t cols);
void matMul64(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols, size_t bCols);
void matMulVec64(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols);

inline void matrixTranspose(Elem32* out, const Elem32* in, size_t rows, size_t cols) {
    matrixTranspose32(out, in, rows, cols);
}
inline void matMul(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bCols) {
    matMul32(out, a, b, aRows, aCols, bCols);
}
inline void matMulTransposedPacked(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bRows, size_t bCols) {
    matMulTransposedPacked32(out, a, b, aRows, aCols, bRows, bCols);
}
inline void matMulVec(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols) {
    matMulVec32(out, a, b, aRows, aCols);
}
inline void matMulVecPacked(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols) {
    matMulVecPacked32(out, a, b, aRows, aCols);
}

template<typename ElemType>
inline void matrixTransposeT(ElemType* out, const ElemType* in, size_t rows, size_t cols);

template<typename ElemType>
inline void matMulT(ElemType* out, const ElemType* a, const ElemType* b, size_t aRows, size_t aCols, size_t bCols);

template<typename ElemType>
inline void matMulVecT(ElemType* out, const ElemType* a, const ElemType* b, size_t aRows, size_t aCols);

template<>
inline void matrixTransposeT<Elem32>(Elem32* out, const Elem32* in, size_t rows, size_t cols) {
    matrixTranspose32(out, in, rows, cols);
}
template<>
inline void matrixTransposeT<Elem64>(Elem64* out, const Elem64* in, size_t rows, size_t cols) {
    matrixTranspose64(out, in, rows, cols);
}

template<>
inline void matMulT<Elem32>(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bCols) {
    matMul32(out, a, b, aRows, aCols, bCols);
}
template<>
inline void matMulT<Elem64>(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols, size_t bCols) {
    matMul64(out, a, b, aRows, aCols, bCols);
}

template<>
inline void matMulVecT<Elem32>(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols) {
    matMulVec32(out, a, b, aRows, aCols);
}
template<>
inline void matMulVecT<Elem64>(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols) {
    matMulVec64(out, a, b, aRows, aCols);
}

}
#endif

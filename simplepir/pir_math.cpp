/**
 * @file pir_math.cpp
 * @brief PIR 矩阵运算实现 (OpenMP + AVX2 SIMD 优化版本)
 *
 * 优化策略：
 * 1. OpenMP 多线程并行化
 * 2. AVX2 SIMD 向量化 (一次处理 8 个 uint32_t)
 *
 * 32-bit 版本: AVX2 优化
 * 64-bit 版本: 标量实现 (用于 embedding PIR with logQ=64)
 */

#include "pir_math.h"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace simplepir {

// ============================================
// 32-bit versions (AVX2 optimized)
// ============================================

#ifdef __AVX2__
inline uint32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return static_cast<uint32_t>(_mm_cvtsi128_si32(sum128));
}
#endif

void matrixTranspose32(Elem32* out, const Elem32* in, size_t rows, size_t cols) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

void matMul32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols, size_t bCols) {
#ifdef __AVX2__
    // 内积路径：沿 aCols 向量化，批量加载 A 行数据并复用给所有 bCols 列
    static constexpr size_t MATMUL32_INNER_MAX = 32;
    if (bCols <= MATMUL32_INNER_MAX) {
        // 转置 b: (aCols × bCols) → bT: (bCols × aCols)
        std::vector<Elem32> bT(bCols * aCols);
        for (size_t k = 0; k < aCols; k++)
            for (size_t j = 0; j < bCols; j++)
                bT[j * aCols + k] = b[k * bCols + j];

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < aRows; i++) {
            const Elem32* aRow = a + aCols * i;
            Elem32* outRow = out + bCols * i;

            __m256i sums[MATMUL32_INNER_MAX];
            for (size_t j = 0; j < bCols; j++)
                sums[j] = _mm256_setzero_si256();

            size_t k = 0;
            for (; k + 8 <= aCols; k += 8) {
                __m256i aVec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(aRow + k));

                for (size_t j = 0; j < bCols; j++) {
                    __m256i bVec = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(bT.data() + j * aCols + k));
                    sums[j] = _mm256_add_epi32(sums[j],
                        _mm256_mullo_epi32(aVec, bVec));
                }
            }

            for (size_t j = 0; j < bCols; j++) {
                Elem32 tmp = hsum_epi32_avx2(sums[j]);
                for (size_t kk = k; kk < aCols; kk++)
                    tmp += aRow[kk] * bT[j * aCols + kk];
                outRow[j] = tmp;
            }
        }
        return;
    }
#endif

    // 回退路径（外积）：bCols 极大时沿 bCols 向量化
    // 注意：对于 PIR 场景 bCols 通常远小于阈值，此路径基本不会执行
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < aRows; i++) {
        Elem32* outRow = out + bCols * i;
        for (size_t j = 0; j < bCols; j++) {
            outRow[j] = 0;
        }

        for (size_t k = 0; k < aCols; k++) {
            Elem32 aik = a[aCols * i + k];
            const Elem32* bRow = b + bCols * k;

#ifdef __AVX2__
            __m256i va = _mm256_set1_epi32(static_cast<int32_t>(aik));
            size_t j = 0;

            for (; j + 8 <= bCols; j += 8) {
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bRow + j));
                __m256i vout = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(outRow + j));
                __m256i vprod = _mm256_mullo_epi32(va, vb);
                vout = _mm256_add_epi32(vout, vprod);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(outRow + j), vout);
            }

            for (; j < bCols; j++) {
                outRow[j] += aik * bRow[j];
            }
#else
            for (size_t j = 0; j < bCols; j++) {
                outRow[j] += aik * bRow[j];
            }
#endif
        }
    }
}

void matMulTransposedPacked32(Elem32* out, const Elem32* a, const Elem32* b,
                               size_t aRows, size_t aCols, size_t bRows, size_t bCols) {
    if (aRows > aCols) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < aRows; i++) {
            size_t ind1 = i * aCols;
            for (size_t k = 0; k < aCols; k++) {
                Elem32 db = a[ind1++];
                Elem32 val = db & MASK32;
                Elem32 val2 = (db >> BASIS) & MASK32;
                Elem32 val3 = (db >> BASIS2) & MASK32;
                for (size_t j = 0; j < bRows; j++) {
                    out[bRows * i + j] += val * b[k * COMPRESSION + j * bCols];
                    out[bRows * i + j] += val2 * b[k * COMPRESSION + j * bCols + 1];
                    out[bRows * i + j] += val3 * b[k * COMPRESSION + j * bCols + 2];
                }
            }
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t j = 0; j < bRows; j += 8) {
            for (size_t i = 0; i < aRows; i++) {
                size_t ind1 = i * aCols;
                Elem32 tmp[8] = {0};
                size_t ind2 = 0;
                for (size_t k = 0; k < aCols; k++) {
                    Elem32 db = a[ind1++];
                    for (int m = 0; m < 3; m++) {
                        Elem32 val = (db >> (m * BASIS)) & MASK32;
                        for (int t = 0; t < 8; t++) {
                            tmp[t] += val * b[ind2 + (j + t) * bCols];
                        }
                        ind2++;
                    }
                }
                for (int t = 0; t < 8; t++) {
                    out[bRows * i + j + t] = tmp[t];
                }
            }
        }
    }
}

void matMulVec32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < aRows; i++) {
        const Elem32* rowPtr = a + aCols * i;

#ifdef __AVX2__
        __m256i sum = _mm256_setzero_si256();
        size_t j = 0;

        for (; j + 8 <= aCols; j += 8) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowPtr + j));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
            __m256i vprod = _mm256_mullo_epi32(va, vb);
            sum = _mm256_add_epi32(sum, vprod);
        }

        Elem32 tmp = hsum_epi32_avx2(sum);

        for (; j < aCols; j++) {
            tmp += rowPtr[j] * b[j];
        }

        out[i] = tmp;
#else
        Elem32 tmp = 0;
        for (size_t j = 0; j < aCols; j++) {
            tmp += rowPtr[j] * b[j];
        }
        out[i] = tmp;
#endif
    }
}

void matMulVecPacked32(Elem32* out, const Elem32* a, const Elem32* b, size_t aRows, size_t aCols) {
    size_t numBlocks = (aRows + 7) / 8;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t block = 0; block < numBlocks; block++) {
        size_t i = block * 8;
        size_t index = i * aCols;
        Elem32 tmp[8] = {0};

        for (size_t j = 0; j < aCols; j++) {
            Elem32 db[8];
            for (int t = 0; t < 8 && (i + t) < aRows; t++) {
                db[t] = a[index + t * aCols];
            }

            size_t index2 = j * COMPRESSION;
            for (int m = 0; m < 3; m++) {
                for (int t = 0; t < 8 && (i + t) < aRows; t++) {
                    tmp[t] += ((db[t] >> (m * BASIS)) & MASK32) * b[index2];
                }
                index2++;
            }
            index++;
        }

        for (int t = 0; t < 8 && (i + t) < aRows; t++) {
            out[i + t] += tmp[t];
        }
    }
}

// ============================================
// 压缩存储矩阵乘法 (DB: uint8_t, Query: uint32_t)
// ============================================

void matMulVecCompressed8_32(Elem32* out, const uint8_t* db, const Elem32* query,
                              size_t dbRows, size_t dbCols, bool signedData) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < dbRows; i++) {
        const uint8_t* rowPtr = db + dbCols * i;
#ifdef __AVX2__
        __m256i sum = _mm256_setzero_si256();
        size_t j = 0;
        for (; j + 8 <= dbCols; j += 8) {
            __m128i db8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowPtr + j));
            // 有符号数据: int8 符号扩展; 无符号数据: uint8 零扩展
            __m256i dbVec = signedData ? _mm256_cvtepi8_epi32(db8)
                                      : _mm256_cvtepu8_epi32(db8);
            __m256i qVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + j));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(dbVec, qVec));
        }
        Elem32 tmp = hsum_epi32_avx2(sum);
        for (; j < dbCols; j++) {
            Elem32 dbVal = signedData ? static_cast<Elem32>(static_cast<int32_t>(static_cast<int8_t>(rowPtr[j])))
                                      : static_cast<Elem32>(rowPtr[j]);
            tmp += dbVal * query[j];
        }
        out[i] = tmp;
#else
        Elem32 tmp = 0;
        for (size_t j = 0; j < dbCols; j++) {
            Elem32 dbVal = signedData ? static_cast<Elem32>(static_cast<int32_t>(static_cast<int8_t>(rowPtr[j])))
                                      : static_cast<Elem32>(rowPtr[j]);
            tmp += dbVal * query[j];
        }
        out[i] = tmp;
#endif
    }
}

void matMulCompressed8_32(Elem32* out, const uint8_t* db, const Elem32* b,
                           size_t dbRows, size_t dbCols, size_t bCols, bool signedData) {
#ifdef __AVX2__
    // 内积路径：沿 dbCols 向量化，批量加载 8 字节 uint8 DB 并复用给所有查询
    // 对压缩 DB 始终优于外积路径（避免逐字节加载 + 读改写输出）
    static constexpr size_t COMPRESSED_INNER_MAX = 64;
    if (bCols <= COMPRESSED_INNER_MAX) {
        std::vector<Elem32> bT(bCols * dbCols);
        for (size_t k = 0; k < dbCols; k++)
            for (size_t j = 0; j < bCols; j++)
                bT[j * dbCols + k] = b[k * bCols + j];

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < dbRows; i++) {
            const uint8_t* rowPtr = db + dbCols * i;
            Elem32* outRow = out + bCols * i;

            __m256i sums[COMPRESSED_INNER_MAX];
            for (size_t j = 0; j < bCols; j++)
                sums[j] = _mm256_setzero_si256();

            size_t k = 0;
            for (; k + 8 <= dbCols; k += 8) {
                __m128i db8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(rowPtr + k));
                // 有符号数据: int8 符号扩展; 无符号数据: uint8 零扩展
                __m256i dbVec = signedData ? _mm256_cvtepi8_epi32(db8)
                                          : _mm256_cvtepu8_epi32(db8);

                for (size_t j = 0; j < bCols; j++) {
                    __m256i bVec = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(bT.data() + j * dbCols + k));
                    sums[j] = _mm256_add_epi32(sums[j],
                        _mm256_mullo_epi32(dbVec, bVec));
                }
            }

            // 水平求和 + 处理 dbCols 尾部
            for (size_t j = 0; j < bCols; j++) {
                Elem32 tmp = hsum_epi32_avx2(sums[j]);
                for (size_t kk = k; kk < dbCols; kk++) {
                    Elem32 dbVal = signedData ? static_cast<Elem32>(static_cast<int32_t>(static_cast<int8_t>(rowPtr[kk])))
                                              : static_cast<Elem32>(rowPtr[kk]);
                    tmp += dbVal * bT[j * dbCols + kk];
                }
                outRow[j] = tmp;
            }
        }
        return;
    }
#endif

    // 回退路径（外积）：bCols 极大时沿 bCols 向量化
    // 注意：对于 PIR 场景 bCols 通常远小于阈值，此路径基本不会执行
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < dbRows; i++) {
        Elem32* outRow = out + bCols * i;
        for (size_t j = 0; j < bCols; j++) outRow[j] = 0;

        for (size_t k = 0; k < dbCols; k++) {
            Elem32 db_ik = signedData ? static_cast<Elem32>(static_cast<int32_t>(static_cast<int8_t>(db[dbCols * i + k])))
                                      : static_cast<Elem32>(db[dbCols * i + k]);
            const Elem32* bRow = b + bCols * k;
#ifdef __AVX2__
            __m256i va = _mm256_set1_epi32(static_cast<int32_t>(db_ik));
            size_t j = 0;
            for (; j + 8 <= bCols; j += 8) {
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bRow + j));
                __m256i vout = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(outRow + j));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(outRow + j),
                    _mm256_add_epi32(vout, _mm256_mullo_epi32(va, vb)));
            }
            for (; j < bCols; j++) outRow[j] += db_ik * bRow[j];
#else
            for (size_t j = 0; j < bCols; j++) outRow[j] += db_ik * bRow[j];
#endif
        }
    }
}

// ============================================
// 64-bit versions (AVX2 optimized for embedding PIR)
// ============================================

#ifdef __AVX2__
// 水平求和: 将 __m256i 中的 4 个 64-bit 值相加
inline uint64_t hsum_epi64_avx2(__m256i v) {
    // 提取高低 128-bit
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    // 相加得到 2 个 64-bit 值
    __m128i sum128 = _mm_add_epi64(lo, hi);
    // 提取两个 64-bit 值并相加
    uint64_t lo64 = static_cast<uint64_t>(_mm_cvtsi128_si64(sum128));
    uint64_t hi64 = static_cast<uint64_t>(_mm_extract_epi64(sum128, 1));
    return lo64 + hi64;
}
#endif

void matrixTranspose64(Elem64* out, const Elem64* in, size_t rows, size_t cols) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

void matMul64(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols, size_t bCols) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < aRows; i++) {
        Elem64* outRow = out + bCols * i;
        for (size_t j = 0; j < bCols; j++) {
            outRow[j] = 0;
        }

        for (size_t k = 0; k < aCols; k++) {
            Elem64 aik = a[aCols * i + k];
            const Elem64* bRow = b + bCols * k;

#ifdef __AVX2__
            __m256i va = _mm256_set1_epi64x(static_cast<int64_t>(aik));
            size_t j = 0;

            // AVX2: 一次处理 4 个 64-bit 值
            for (; j + 4 <= bCols; j += 4) {
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bRow + j));
                __m256i vout = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(outRow + j));
                // 64-bit 乘法: _mm256_mullo_epi64 需要 AVX-512，用手动实现
                // 拆分为 32-bit 乘法
                __m256i blo = _mm256_and_si256(vb, _mm256_set1_epi64x(0xFFFFFFFF));
                __m256i bhi = _mm256_srli_epi64(vb, 32);
                __m256i alo = _mm256_and_si256(va, _mm256_set1_epi64x(0xFFFFFFFF));
                __m256i ahi = _mm256_srli_epi64(va, 32);
                // prod = a*b = (alo + ahi<<32) * (blo + bhi<<32)
                //            = alo*blo + (alo*bhi + ahi*blo)<<32  (忽略高64位溢出)
                __m256i prod_lo = _mm256_mul_epu32(alo, blo);
                __m256i prod_mid1 = _mm256_mul_epu32(alo, bhi);
                __m256i prod_mid2 = _mm256_mul_epu32(ahi, blo);
                __m256i prod_mid = _mm256_add_epi64(prod_mid1, prod_mid2);
                prod_mid = _mm256_slli_epi64(prod_mid, 32);
                __m256i vprod = _mm256_add_epi64(prod_lo, prod_mid);
                vout = _mm256_add_epi64(vout, vprod);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(outRow + j), vout);
            }

            // 处理剩余元素
            for (; j < bCols; j++) {
                outRow[j] += aik * bRow[j];
            }
#else
            for (size_t j = 0; j < bCols; j++) {
                outRow[j] += aik * bRow[j];
            }
#endif
        }
    }
}

void matMulVec64(Elem64* out, const Elem64* a, const Elem64* b, size_t aRows, size_t aCols) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < aRows; i++) {
        const Elem64* rowPtr = a + aCols * i;

#ifdef __AVX2__
        __m256i sum = _mm256_setzero_si256();
        size_t j = 0;

        // AVX2: 一次处理 4 个 64-bit 值
        for (; j + 4 <= aCols; j += 4) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowPtr + j));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));

            // 64-bit 乘法实现 (AVX2 没有原生 64-bit 乘法)
            __m256i alo = _mm256_and_si256(va, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i ahi = _mm256_srli_epi64(va, 32);
            __m256i blo = _mm256_and_si256(vb, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i bhi = _mm256_srli_epi64(vb, 32);

            // prod = alo*blo + (alo*bhi + ahi*blo)<<32
            __m256i prod_lo = _mm256_mul_epu32(alo, blo);
            __m256i prod_mid1 = _mm256_mul_epu32(alo, bhi);
            __m256i prod_mid2 = _mm256_mul_epu32(ahi, blo);
            __m256i prod_mid = _mm256_add_epi64(prod_mid1, prod_mid2);
            prod_mid = _mm256_slli_epi64(prod_mid, 32);
            __m256i vprod = _mm256_add_epi64(prod_lo, prod_mid);

            sum = _mm256_add_epi64(sum, vprod);
        }

        Elem64 tmp = hsum_epi64_avx2(sum);

        // 处理剩余元素
        for (; j < aCols; j++) {
            tmp += rowPtr[j] * b[j];
        }

        out[i] = tmp;
#else
        Elem64 tmp = 0;
        for (size_t j = 0; j < aCols; j++) {
            tmp += rowPtr[j] * b[j];
        }
        out[i] = tmp;
#endif
    }
}

} // namespace simplepir

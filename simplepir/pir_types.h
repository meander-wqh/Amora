/**
 * @file pir_types.h
 * @brief SimplePIR 基本类型定义
 */

#ifndef PIR_TYPES_H
#define PIR_TYPES_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace simplepir {

// Element types for different PIR configurations
using Elem32 = uint32_t;  // For logQ <= 32 (neighbor PIR)
using Elem64 = uint64_t;  // For logQ <= 64 (embedding PIR)

// Default Elem type for backward compatibility (neighbor PIR uses 32-bit)
using Elem = Elem32;

using PRGKey = std::array<uint8_t, 16>;

constexpr uint64_t COMPRESSION = 3;
constexpr uint64_t BASIS = 10;
constexpr uint64_t BASIS2 = BASIS * 2;
constexpr Elem32 MASK32 = (1 << BASIS) - 1;
constexpr Elem64 MASK64 = (1ULL << BASIS) - 1;

struct Params {
    uint64_t N = 0;
    double Sigma = 0.0;
    uint64_t L = 0;
    uint64_t M = 0;
    uint64_t Logq = 0;
    uint64_t P = 0;
    
    uint64_t delta() const {
        if (Logq >= 64) {
            // Use 128-bit arithmetic to avoid overflow when logQ >= 64
            __uint128_t q = (__uint128_t)1 << Logq;
            return static_cast<uint64_t>(q / P);
        }
        return (1ULL << Logq) / P;
    }
    uint64_t deltaLogScale() const {
        return static_cast<uint64_t>(std::ceil(static_cast<double>(Logq) / std::log2(static_cast<double>(P))));
    }
    uint64_t round(uint64_t x) const {
        uint64_t d = delta();
        uint64_t v = (x + d / 2) / d;
        return v % P;
    }
    void pickParams(bool doublePir, uint64_t numSamples);
    void printParams() const;

    // Get element size in bytes based on logQ
    uint64_t elemSize() const {
        return (Logq <= 32) ? sizeof(Elem32) : sizeof(Elem64);
    }
};

/**
 * @brief 数据库信息结构体，存储 PIR 数据库的元数据
 * 
 * SimplePIR 将数据组织成 L×M 矩阵，查询返回整列数据
 */
struct DBInfo {
    uint64_t Num = 0;        // 数据库中原始元素的数量
    uint64_t RowLength = 0;  // 每个元素的比特长度
    uint64_t Packing = 0;    // 打包因子：每个矩阵元素可存储多少个原始元素（当元素较小时）
    uint64_t Ne = 0;         // 编码扩展因子：每个原始元素需要多少个矩阵行来编码（base-P 编码）
    uint64_t X = 0;          // 保留字段
    uint64_t P = 0;          // 明文模数，用于 LWE 加密的模 P 运算
    uint64_t Logq = 0;       // 密文模数的比特数，q = 2^Logq
    uint64_t Basis = 0;      // 压缩基数，用于数据库压缩存储，每个小元素占用的比特数，默认为 10
    uint64_t Squishing = 0;  // 压缩因子，减少存储和通信开销，压缩因子（几个元素打包成一个）, Squishing = 1 表示不压缩
    uint64_t Cols = 0;       // 数据库矩阵的实际列数
    uint64_t L = 0;          // 数据库矩阵的行数（物理行 = L，逻辑行 = L/Ne）
    uint64_t M = 0;          // 数据库矩阵的列数（一次查询选择一列，返回 L 行数据）
};

// Forward declarations for templated Matrix
template<typename ElemType> class MatrixT;
using Matrix = MatrixT<Elem32>;      // Default 32-bit matrix
using Matrix64 = MatrixT<Elem64>;    // 64-bit matrix for embedding PIR

class Database;

struct State {
    std::vector<std::shared_ptr<Matrix>> data;
    State() = default;
    explicit State(std::shared_ptr<Matrix> m) { data.push_back(m); }
    State(std::initializer_list<std::shared_ptr<Matrix>> matrices) : data(matrices) {}
};

struct CompressedState {
    PRGKey seed;
};

struct Msg {
    std::vector<std::shared_ptr<Matrix>> data;
    Msg() = default;
    explicit Msg(std::shared_ptr<Matrix> m) { data.push_back(m); }
    Msg(std::initializer_list<std::shared_ptr<Matrix>> matrices) : data(matrices) {}
    uint64_t size() const;
    uint64_t sizeBytes() const;  // Actual bytes for communication cost
};

struct MsgSlice {
    std::vector<Msg> data;
    MsgSlice() = default;
    explicit MsgSlice(const Msg& m) { data.push_back(m); }
    uint64_t size() const;
    uint64_t sizeBytes() const;  // Actual bytes for communication cost
};

// baseP: 返回单个数字的第 i 位（以 p 为基数）
inline uint64_t basePDigit(uint64_t p, uint64_t m, uint64_t i) {
    for (uint64_t j = 0; j < i; j++) m = m / p;
    return m % p;
}

// baseP: 返回数字的所有位（以 p 为基数，共 numDigits 位）
inline std::vector<uint64_t> baseP(uint64_t p, uint64_t m, uint64_t numDigits) {
    std::vector<uint64_t> result(numDigits);
    for (uint64_t i = 0; i < numDigits; i++) {
        result[i] = m % p;
        m /= p;
    }
    return result;
}

inline uint64_t reconstructFromBaseP(uint64_t p, const std::vector<uint64_t>& vals) {
    uint64_t res = 0, coeff = 1;
    for (uint64_t v : vals) { res += coeff * v; coeff *= p; }
    return res;
}

inline uint64_t computeNumEntriesBaseP(uint64_t p, uint64_t logQ) {
    return static_cast<uint64_t>(std::ceil(static_cast<double>(logQ) / std::log2(static_cast<double>(p))));
}

std::tuple<uint64_t, uint64_t, uint64_t> numDBEntries(uint64_t N, uint64_t rowLength, uint64_t p);
uint64_t reconstructElem(std::vector<uint64_t>& vals, uint64_t index, const DBInfo& info);
std::pair<uint64_t, uint64_t> approxSquareDatabaseDims(uint64_t N, uint64_t rowLength, uint64_t p);

} // namespace simplepir

#endif // PIR_TYPES_H

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

using Elem32 = uint32_t;
using Elem64 = uint64_t;

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

    uint64_t elemSize() const {
        return (Logq <= 32) ? sizeof(Elem32) : sizeof(Elem64);
    }
};

struct DBInfo {
    uint64_t Num = 0;
    uint64_t RowLength = 0;
    uint64_t Packing = 0;
    uint64_t Ne = 0;
    uint64_t X = 0;
    uint64_t P = 0;
    uint64_t Logq = 0;
    uint64_t Basis = 0;
    uint64_t Squishing = 0;
    uint64_t Cols = 0;
    uint64_t L = 0;
    uint64_t M = 0;
};

template<typename ElemType> class MatrixT;
using Matrix = MatrixT<Elem32>;
using Matrix64 = MatrixT<Elem64>;

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
    uint64_t sizeBytes() const;
};

struct MsgSlice {
    std::vector<Msg> data;
    MsgSlice() = default;
    explicit MsgSlice(const Msg& m) { data.push_back(m); }
    uint64_t size() const;
    uint64_t sizeBytes() const;
};

inline uint64_t basePDigit(uint64_t p, uint64_t m, uint64_t i) {
    for (uint64_t j = 0; j < i; j++) m = m / p;
    return m % p;
}

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

}

#endif

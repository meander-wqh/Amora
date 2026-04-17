#include "pir_types.h"
#include "matrix.h"
#include <iostream>
#include <cmath>

namespace simplepir {

struct ParamEntry { uint64_t logN, logM, logQ; double sigma; uint64_t pSimple, pDouble; };
static const std::vector<ParamEntry> LWE_PARAMS = {{10, 13, 32, 6.4, 991, 929}};

void Params::pickParams(bool doublePir, uint64_t numSamples) {
    if (N == 0 || Logq == 0) throw std::runtime_error("Need to specify n and q!");
    for (const auto& e : LWE_PARAMS) {
        if (N == (1ULL << e.logN) && numSamples <= (1ULL << e.logM) && Logq == e.logQ) {
            Sigma = e.sigma; P = doublePir ? e.pDouble : e.pSimple;
            if (Sigma == 0.0 || P == 0) throw std::runtime_error("Params invalid!");
            return;
        }
    }
    Sigma = 6.4; P = 991;
}

void Params::printParams() const {
    std::cout << "Working with: n=" << N << "; db size=2^" << int(std::log2(L) + std::log2(M))
              << " (l=" << L << ", m=" << M << "); logq=" << Logq << "; p=" << P << "; sigma=" << Sigma << std::endl;
}

uint64_t Msg::size() const {
    uint64_t sz = 0;
    for (const auto& d : data) if (d) sz += d->size();
    return sz;
}

uint64_t Msg::sizeBytes() const {
    uint64_t sz = 0;
    for (const auto& d : data) if (d) sz += d->sizeBytes();
    return sz;
}

uint64_t MsgSlice::size() const {
    uint64_t sz = 0;
    for (const auto& d : data) sz += d.size();
    return sz;
}

uint64_t MsgSlice::sizeBytes() const {
    uint64_t sz = 0;
    for (const auto& d : data) sz += d.sizeBytes();
    return sz;
}

std::tuple<uint64_t, uint64_t, uint64_t> numDBEntries(uint64_t N, uint64_t rowLength, uint64_t p) {
    double logP = std::log2(static_cast<double>(p));
    if (static_cast<double>(rowLength) <= logP) {
        uint64_t entriesPerElem = static_cast<uint64_t>(logP) / rowLength;
        uint64_t dbEntries = (N + entriesPerElem - 1) / entriesPerElem;
        return {dbEntries, 1, entriesPerElem};
    }
    uint64_t ne = computeNumEntriesBaseP(p, rowLength);
    return {N * ne, ne, 0};
}

uint64_t reconstructElem(std::vector<uint64_t>& vals, uint64_t index, const DBInfo& info) {
    uint64_t q = 1ULL << info.Logq;
    for (auto& v : vals) { v = (v + info.P / 2) % q; v = v % info.P; }
    uint64_t val = reconstructFromBaseP(info.P, vals);
    if (info.Packing > 0) val = basePDigit(1ULL << info.RowLength, val, index % info.Packing);
    return val;
}

std::pair<uint64_t, uint64_t> approxSquareDatabaseDims(uint64_t N, uint64_t rowLength, uint64_t p) {
    auto [dbElems, elemsPerEntry, unused] = numDBEntries(N, rowLength, p);
    uint64_t l = static_cast<uint64_t>(std::sqrt(static_cast<double>(dbElems)));
    uint64_t rem = l % elemsPerEntry;
    if (rem != 0) l += elemsPerEntry - rem;
    uint64_t m = (dbElems + l - 1) / l;
    return {l, m};
}

} // namespace simplepir

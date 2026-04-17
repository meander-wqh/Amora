#include "random.h"
#include <cstring>
namespace simplepir {

const std::vector<double> GaussSampler::cdfTable_ = {
    0.5, 0.987867, 0.952345, 0.895957, 0.822578, 0.736994, 0.644389, 0.549831, 0.457833, 0.372034,
    0.295023, 0.22831, 0.172422, 0.127074, 0.0913938, 0.0641467, 0.0439369, 0.0293685, 0.0191572,
    0.0121949, 0.00757568, 0.00459264, 0.00271706, 0.00156868, 0.000883826, 0.000485955, 0.000260749,
    0.000136536, 6.97696e-05, 3.47923e-05, 1.69316e-05, 8.041e-06, 3.72665e-06, 1.68549e-06
};

constexpr size_t BUF_SIZE = 1024;

PRGReader::PRGReader() {
    std::random_device rd;
    for (size_t i = 0; i < key_.size(); i += sizeof(unsigned int)) {
        unsigned int r = rd();
        std::memcpy(key_.data() + i, &r, std::min(sizeof(unsigned int), key_.size() - i));
    }
    uint64_t seed = 0;
    std::memcpy(&seed, key_.data(), sizeof(seed));
    rng_.seed(seed);
}

PRGReader::PRGReader(const PRGKey& key) : key_(key) {
    uint64_t seed = 0;
    std::memcpy(&seed, key_.data(), sizeof(seed));
    rng_.seed(seed);
}

uint64_t PRGReader::uint64() { return rng_(); }
uint64_t PRGReader::randMod(uint64_t mod) { return mod ? rng_() % mod : 0; }

BufPRGReader::BufPRGReader() : prg_(std::make_shared<PRGReader>()), buffer_(BUF_SIZE), bufferIndex_(BUF_SIZE) {}
BufPRGReader::BufPRGReader(std::shared_ptr<PRGReader> prg) : prg_(prg), buffer_(BUF_SIZE), bufferIndex_(BUF_SIZE) {}

void BufPRGReader::refillBuffer() {
    for (auto& v : buffer_) v = prg_->uint64();
    bufferIndex_ = 0;
}

uint64_t BufPRGReader::uint64() {
    if (bufferIndex_ >= buffer_.size()) refillBuffer();
    return buffer_[bufferIndex_++];
}

int64_t BufPRGReader::int63() { return static_cast<int64_t>(uint64() % (1ULL << 63)); }

uint64_t BufPRGReader::randMod(uint64_t mod) {
    if (!mod) return 0;
    uint64_t max = UINT64_MAX - (UINT64_MAX % mod);
    uint64_t r;
    do { r = uint64(); } while (r >= max);
    return r % mod;
}

double BufPRGReader::float64() { return static_cast<double>(uint64()) / static_cast<double>(UINT64_MAX); }
int BufPRGReader::intn(int n) { return n > 0 ? static_cast<int>(randMod(static_cast<uint64_t>(n))) : 0; }

int64_t GaussSampler::sample() { return sample(globalPRG()); }
int64_t GaussSampler::sample(BufPRGReader& prg) {
    int64_t x; double y;
    do { x = prg.intn(static_cast<int>(cdfTable_.size())); y = prg.float64(); } while (y >= cdfTable_[x]);
    return prg.uint64() % 2 == 0 ? -x : x;
}

namespace {
    std::mutex globalPRGMutex;
    std::unique_ptr<BufPRGReader> globalPRGInstance;
    BufPRGReader& getOrCreateGlobalPRG() {
        if (!globalPRGInstance) globalPRGInstance = std::make_unique<BufPRGReader>();
        return *globalPRGInstance;
    }
}

PRGKey randomPRGKey() {
    PRGKey key;
    std::random_device rd;
    for (size_t i = 0; i < key.size(); i += sizeof(unsigned int)) {
        unsigned int r = rd();
        std::memcpy(key.data() + i, &r, std::min(sizeof(unsigned int), key.size() - i));
    }
    return key;
}

BufPRGReader& globalPRG() { std::lock_guard<std::mutex> lock(globalPRGMutex); return getOrCreateGlobalPRG(); }
void setGlobalPRG(const PRGKey& key) {
    std::lock_guard<std::mutex> lock(globalPRGMutex);
    globalPRGInstance = std::make_unique<BufPRGReader>(std::make_shared<PRGReader>(key));
}

// 使用线程局部 PRG 避免锁竞争
thread_local std::unique_ptr<BufPRGReader> threadLocalPRG;

BufPRGReader& getThreadLocalPRG() {
    if (!threadLocalPRG) {
        // 每个线程创建独立的 PRG
        threadLocalPRG = std::make_unique<BufPRGReader>();
    }
    return *threadLocalPRG;
}

uint64_t randInt(uint64_t mod) {
    return getThreadLocalPRG().randMod(mod);
}

int64_t gaussSample() {
    return GaussSampler::sample(getThreadLocalPRG());
}

} // namespace simplepir

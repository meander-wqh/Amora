#ifndef PIR_RANDOM_H
#define PIR_RANDOM_H
#include "pir_types.h"
#include <random>
#include <mutex>
#include <memory>
namespace simplepir {

class PRGReader {
public:
    PRGReader();
    explicit PRGReader(const PRGKey& key);
    uint64_t uint64();
    uint64_t randMod(uint64_t mod);
    const PRGKey& key() const { return key_; }
private:
    PRGKey key_;
    std::mt19937_64 rng_;
};

class BufPRGReader {
public:
    BufPRGReader();
    explicit BufPRGReader(std::shared_ptr<PRGReader> prg);
    uint64_t uint64();
    int64_t int63();
    uint64_t randMod(uint64_t mod);
    double float64();
    int intn(int n);
    const PRGKey& key() const { return prg_->key(); }
private:
    std::shared_ptr<PRGReader> prg_;
    std::vector<uint64_t> buffer_;
    size_t bufferIndex_;
    void refillBuffer();
};

class GaussSampler {
public:
    static int64_t sample();
    static int64_t sample(BufPRGReader& prg);
private:
    static const std::vector<double> cdfTable_;
};

PRGKey randomPRGKey();
BufPRGReader& globalPRG();
void setGlobalPRG(const PRGKey& key);
uint64_t randInt(uint64_t mod);
int64_t gaussSample();

} // namespace simplepir
#endif // PIR_RANDOM_H

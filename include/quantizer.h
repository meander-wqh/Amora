#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>

namespace hnsw {

struct QuantizerConfig {
    int bits = 8;
    float scale = 256.0f;
    bool isUnsigned = true;
    bool normalize = true;  // 量化前是否归一化向量
    
    void setFromBits(int b, bool unsignedMode = true) {
        bits = b;
        isUnsigned = unsignedMode;
        if (unsignedMode) scale = static_cast<float>(1 << (bits + 1));
        else scale = static_cast<float>(1 << bits);
    }
    
    static QuantizerConfig tiptoeDefault() {
        QuantizerConfig cfg;
        cfg.bits = 5; cfg.scale = 32.0f; cfg.isUnsigned = false;
        return cfg;
    }
    
    static QuantizerConfig uint8Default() {
        QuantizerConfig cfg;
        cfg.bits = 8; cfg.scale = 512.0f; cfg.isUnsigned = true;
        return cfg;
    }
};

class ScalarQuantizer {
public:
    QuantizerConfig config;
    int dim;
    int minVal, maxVal;
    float scale;  // 实际使用的scale（训练后可能调整）
    bool trained;
    
    ScalarQuantizer() : dim(0), minVal(0), maxVal(0), scale(1.0f), trained(false) {}
    
    // 计算向量L2范数
    static float computeNorm(const float* vec, int dim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum + 1e-10f);  // 避免除零
    }
    
    // 归一化单个向量（原地）
    static void normalizeVector(float* vec, int dim) {
        float norm = computeNorm(vec, dim);
        float invNorm = 1.0f / norm;
        for (int i = 0; i < dim; ++i) {
            vec[i] *= invNorm;
        }
    }
    
    // 归一化向量数组（返回新数组）
    static std::vector<float> normalizeVectors(const float* data, int n, int dim) {
        std::vector<float> normalized(static_cast<size_t>(n) * dim);
        for (int i = 0; i < n; ++i) {
            const float* src = data + static_cast<size_t>(i) * dim;
            float* dst = normalized.data() + static_cast<size_t>(i) * dim;
            float norm = computeNorm(src, dim);
            float invNorm = 1.0f / norm;
            for (int j = 0; j < dim; ++j) {
                dst[j] = src[j] * invNorm;
            }
        }
        return normalized;
    }
    
    ScalarQuantizer(int dimension, const QuantizerConfig& cfg = QuantizerConfig::uint8Default())
        : config(cfg), dim(dimension), scale(cfg.scale), trained(false) { updateRange(); }
    
    void init(int dimension, const QuantizerConfig& cfg) {
        dim = dimension; config = cfg; scale = cfg.scale; trained = false; updateRange();
    }
    
    void updateRange() {
        if (config.isUnsigned) { minVal = 0; maxVal = (1 << config.bits) - 1; }
        else { minVal = -(1 << (config.bits - 1)); maxVal = (1 << (config.bits - 1)) - 1; }
    }
    
    // 训练量化器：分析数据范围，计算最佳scale
    void train(const float* data, int n) {
        if (n <= 0 || dim <= 0) return;
        
        // 如果需要归一化，先对采样数据归一化后再分析
        std::vector<float> normalizedData;
        const float* analyzeData = data;
        int analyzeN = n;
        
        if (config.normalize) {
            // 归一化后数据范围在 [-1, 1]（或 [0, 1] 对于非负数据）
            // 采样并归一化
            int sampleSize = std::min(n, 10000);
            int step = std::max(1, n / sampleSize);
            int actualSamples = (n + step - 1) / step;
            
            normalizedData.resize(static_cast<size_t>(actualSamples) * dim);
            int idx = 0;
            for (int i = 0; i < n && idx < actualSamples; i += step, ++idx) {
                const float* src = data + static_cast<size_t>(i) * dim;
                float* dst = normalizedData.data() + static_cast<size_t>(idx) * dim;
                float norm = computeNorm(src, dim);
                float invNorm = 1.0f / norm;
                for (int j = 0; j < dim; ++j) {
                    dst[j] = src[j] * invNorm;
                }
            }
            analyzeData = normalizedData.data();
            analyzeN = idx;  // 实际采样数
        }
        
        float dataMin = std::numeric_limits<float>::max();
        float dataMax = std::numeric_limits<float>::lowest();
        
        // 采样分析
        int sampleSize = std::min(analyzeN, 10000);
        int step = config.normalize ? 1 : std::max(1, analyzeN / sampleSize);
        
        for (int i = 0; i < analyzeN; i += step) {
            for (int j = 0; j < dim; ++j) {
                float v = analyzeData[(size_t)i * dim + j];
                dataMin = std::min(dataMin, v);
                dataMax = std::max(dataMax, v);
            }
        }
        
        // 计算scale
        if (config.normalize) {
            // 归一化后数据在 [-1, 1] 或 [0, 1] 范围
            if (dataMin >= 0) {
                // 非负归一化数据：[0, ~1] -> [0, 255]
                scale = maxVal / (dataMax + 1e-6f);
            } else {
                // 包含负值：[-1, 1] -> [0, 255] 或 [-128, 127]
                float absMax = std::max(std::abs(dataMin), std::abs(dataMax));
                if (config.isUnsigned) {
                    scale = maxVal / (2.0f * absMax + 1e-6f);
                } else {
                    scale = maxVal / (absMax + 1e-6f);
                }
            }
            std::cout << "Quantizer trained with normalization: dataRange=[" 
                      << dataMin << ", " << dataMax << "], scale=" << scale << std::endl;
        } else if (config.isUnsigned && dataMin >= 0) {
            // 非负数据映射到 [0, 255]
            scale = (maxVal - minVal) / (dataMax - dataMin + 1e-6f);
        } else {
            // 有符号或包含负值
            float absMax = std::max(std::abs(dataMin), std::abs(dataMax));
            scale = maxVal / (absMax + 1e-6f);
        }
        
        trained = true;
    }
    
    // 量化单个值
    inline int8_t quantizeSigned(float val) const {
        int scaled = static_cast<int>(std::round(val * scale));
        return static_cast<int8_t>(std::max(minVal, std::min(maxVal, scaled)));
    }
    
    inline uint8_t quantizeUnsigned(float val) const {
        int scaled = static_cast<int>(std::round(val * scale));
        return static_cast<uint8_t>(std::max(0, std::min(maxVal, scaled)));
    }
    
    // 量化向量数组（支持归一化，根据 config.isUnsigned 选择量化方式）
    void quantize(const float* data, uint8_t* output, int n) const {
        if (config.normalize) {
            // 归一化后量化
            std::vector<float> tempVec(dim);
            for (int i = 0; i < n; ++i) {
                const float* vec = data + static_cast<size_t>(i) * dim;
                uint8_t* out = output + static_cast<size_t>(i) * dim;

                // 归一化到临时缓冲区
                float norm = computeNorm(vec, dim);
                float invNorm = 1.0f / norm;
                for (int j = 0; j < dim; ++j) {
                    tempVec[j] = vec[j] * invNorm;
                }

                // 根据配置选择量化方式
                if (config.isUnsigned) {
                    for (int j = 0; j < dim; ++j) {
                        out[j] = quantizeUnsigned(tempVec[j]);
                    }
                } else {
                    // 有符号量化，存储为 uint8_t（保留位模式）
                    for (int j = 0; j < dim; ++j) {
                        out[j] = static_cast<uint8_t>(quantizeSigned(tempVec[j]));
                    }
                }
            }
        } else {
            // 直接量化
            for (int i = 0; i < n; ++i) {
                const float* vec = data + static_cast<size_t>(i) * dim;
                uint8_t* out = output + static_cast<size_t>(i) * dim;
                if (config.isUnsigned) {
                    for (int j = 0; j < dim; ++j) {
                        out[j] = quantizeUnsigned(vec[j]);
                    }
                } else {
                    // 有符号量化，存储为 uint8_t（保留位模式）
                    for (int j = 0; j < dim; ++j) {
                        out[j] = static_cast<uint8_t>(quantizeSigned(vec[j]));
                    }
                }
            }
        }
    }
    
    void quantize(const float* data, int8_t* output, int n) const {
        if (config.normalize) {
            // 归一化后量化
            std::vector<float> tempVec(dim);
            for (int i = 0; i < n; ++i) {
                const float* vec = data + static_cast<size_t>(i) * dim;
                int8_t* out = output + static_cast<size_t>(i) * dim;
                
                // 归一化到临时缓冲区
                float norm = computeNorm(vec, dim);
                float invNorm = 1.0f / norm;
                for (int j = 0; j < dim; ++j) {
                    tempVec[j] = vec[j] * invNorm;
                }
                
                // 量化归一化后的向量
                for (int j = 0; j < dim; ++j) {
                    out[j] = quantizeSigned(tempVec[j]);
                }
            }
        } else {
            // 直接量化
            for (int i = 0; i < n; ++i) {
                const float* vec = data + static_cast<size_t>(i) * dim;
                int8_t* out = output + static_cast<size_t>(i) * dim;
                for (int j = 0; j < dim; ++j) {
                    out[j] = quantizeSigned(vec[j]);
                }
            }
        }
    }
    
    std::vector<int8_t> quantizeVectorSigned(const float* vec) const {
        std::vector<int8_t> result(dim);
        if (config.normalize) {
            float norm = computeNorm(vec, dim);
            float invNorm = 1.0f / norm;
            for (int i = 0; i < dim; i++) result[i] = quantizeSigned(vec[i] * invNorm);
        } else {
            for (int i = 0; i < dim; i++) result[i] = quantizeSigned(vec[i]);
        }
        return result;
    }
    
    std::vector<uint8_t> quantizeVectorUnsigned(const float* vec) const {
        std::vector<uint8_t> result(dim);
        if (config.normalize) {
            float norm = computeNorm(vec, dim);
            float invNorm = 1.0f / norm;
            for (int i = 0; i < dim; i++) result[i] = quantizeUnsigned(vec[i] * invNorm);
        } else {
            for (int i = 0; i < dim; i++) result[i] = quantizeUnsigned(vec[i]);
        }
        return result;
    }

    // 统一的量化接口：根据 config.isUnsigned 自动选择量化方式
    std::vector<uint8_t> quantizeVector(const float* vec) const {
        std::vector<uint8_t> result(dim);
        if (config.normalize) {
            float norm = computeNorm(vec, dim);
            float invNorm = 1.0f / norm;
            if (config.isUnsigned) {
                for (int i = 0; i < dim; i++) result[i] = quantizeUnsigned(vec[i] * invNorm);
            } else {
                for (int i = 0; i < dim; i++) result[i] = static_cast<uint8_t>(quantizeSigned(vec[i] * invNorm));
            }
        } else {
            if (config.isUnsigned) {
                for (int i = 0; i < dim; i++) result[i] = quantizeUnsigned(vec[i]);
            } else {
                for (int i = 0; i < dim; i++) result[i] = static_cast<uint8_t>(quantizeSigned(vec[i]));
            }
        }
        return result;
    }
    
    // 序列化
    void save(std::ofstream& ofs) const {
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        ofs.write(reinterpret_cast<const char*>(&config.bits), sizeof(config.bits));
        ofs.write(reinterpret_cast<const char*>(&config.scale), sizeof(config.scale));
        ofs.write(reinterpret_cast<const char*>(&config.isUnsigned), sizeof(config.isUnsigned));
        ofs.write(reinterpret_cast<const char*>(&config.normalize), sizeof(config.normalize));
        ofs.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        ofs.write(reinterpret_cast<const char*>(&trained), sizeof(trained));
    }
    
    void load(std::ifstream& ifs) {
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        ifs.read(reinterpret_cast<char*>(&config.bits), sizeof(config.bits));
        ifs.read(reinterpret_cast<char*>(&config.scale), sizeof(config.scale));
        ifs.read(reinterpret_cast<char*>(&config.isUnsigned), sizeof(config.isUnsigned));
        ifs.read(reinterpret_cast<char*>(&config.normalize), sizeof(config.normalize));
        ifs.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        ifs.read(reinterpret_cast<char*>(&trained), sizeof(trained));
        updateRange();
    }
    
    // 距离计算
    static int64_t dotProductInt8(const int8_t* a, const int8_t* b, int dim) {
        int64_t sum = 0;
        for (int i = 0; i < dim; i++) sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
        return sum;
    }
    
    static uint64_t dotProductUint8(const uint8_t* a, const uint8_t* b, int dim) {
        uint64_t sum = 0;
        for (int i = 0; i < dim; i++) sum += static_cast<uint64_t>(a[i]) * static_cast<uint64_t>(b[i]);
        return sum;
    }
    
    static int64_t l2DistanceSquaredInt8(const int8_t* a, const int8_t* b, int dim) {
        int64_t sum = 0;
        for (int i = 0; i < dim; i++) { int64_t diff = a[i] - b[i]; sum += diff * diff; }
        return sum;
    }
    
    static int64_t l2DistanceSquaredUint8(const uint8_t* a, const uint8_t* b, int dim) {
        int64_t sum = 0;
        for (int i = 0; i < dim; i++) { int64_t diff = a[i] - b[i]; sum += diff * diff; }
        return sum;
    }
    
    void printConfig() const {
        std::cout << "Quantizer: bits=" << config.bits << ", scale=" << scale
                  << ", mode=" << (config.isUnsigned ? "unsigned" : "signed")
                  << ", normalize=" << (config.normalize ? "yes" : "no")
                  << ", range=[" << minVal << "," << maxVal << "], dim=" << dim 
                  << ", trained=" << (trained ? "yes" : "no") << "\n";
    }
};

} // namespace hnsw

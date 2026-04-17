#include "io.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <unordered_set>

namespace hnsw {

std::vector<float> readFvecs(const std::string& filename, int& dim, int& n) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + filename);
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    dim = d;
    n = fileSize / (sizeof(int32_t) + d * sizeof(float));
    std::vector<float> data((size_t)n * d);
    in.seekg(0, std::ios::beg);
    for (int i = 0; i < n; i++) {
        int32_t dim_check;
        in.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
        in.read(reinterpret_cast<char*>(data.data() + (size_t)i * d), d * sizeof(float));
    }
    std::cout << "Read " << n << " vectors (dim=" << d << ") from " << filename << std::endl;
    return data;
}

std::vector<int32_t> readIvecs(const std::string& filename, int& dim, int& n) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + filename);
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    dim = d;
    n = fileSize / (sizeof(int32_t) + d * sizeof(int32_t));
    std::vector<int32_t> data(n * d);
    in.seekg(0, std::ios::beg);
    for (int i = 0; i < n; i++) {
        int32_t dim_check;
        in.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
        in.read(reinterpret_cast<char*>(data.data() + i * d), d * sizeof(int32_t));
    }
    std::cout << "Read " << n << " vectors (dim=" << d << ") from " << filename << std::endl;
    return data;
}

std::vector<float> readBvecs(const std::string& filename, int& dim, int& n) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + filename);
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t d;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    dim = d;
    n = fileSize / (sizeof(int32_t) + d);
    std::vector<float> data((size_t)n * d);
    std::vector<uint8_t> buffer(d);
    in.seekg(0, std::ios::beg);
    for (int i = 0; i < n; i++) {
        int32_t dim_check;
        in.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
        in.read(reinterpret_cast<char*>(buffer.data()), d);
        for (int j = 0; j < d; j++) data[(size_t)i * d + j] = static_cast<float>(buffer[j]);
    }
    std::cout << "Read " << n << " vectors (dim=" << d << ") from " << filename << std::endl;
    return data;
}

void writeFvecs(const std::string& filename, const float* data, int dim, int n) {
    std::ofstream out(filename, std::ios::binary);
    int32_t d = dim;
    for (int i = 0; i < n; i++) {
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(data + i * dim), dim * sizeof(float));
    }
}

float computeRecall(const std::vector<int32_t>& groundTruth, const std::vector<int64_t>& results, int k, int nq) {
    int correct = 0, gtDim = groundTruth.size() / nq, resDim = results.size() / nq;
    for (int i = 0; i < nq; i++) {
        std::unordered_set<int32_t> gt_set;
        for (int j = 0; j < std::min(k, gtDim); j++) gt_set.insert(groundTruth[i * gtDim + j]);
        for (int j = 0; j < std::min(k, resDim); j++) if (gt_set.count(results[i * resDim + j])) correct++;
    }
    return static_cast<float>(correct) / (nq * k);
}

}

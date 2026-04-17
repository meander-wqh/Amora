#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace hnsw {
std::vector<float> readFvecs(const std::string& filename, int& dim, int& n);
std::vector<int32_t> readIvecs(const std::string& filename, int& dim, int& n);
std::vector<float> readBvecs(const std::string& filename, int& dim, int& n);
void writeFvecs(const std::string& filename, const float* data, int dim, int n);
float computeRecall(const std::vector<int32_t>& groundTruth, const std::vector<int64_t>& results, int k, int nq);
}

#include "cpu_baselines.h"
#include "cublas_baseline.cuh"
#include "cuda_check.cuh"
#include "matmul.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using Shape = std::tuple<int, int, int>;
using Kernel = void (*)(const float*, const float*, float*, int, int, int);

std::vector<double> reference(const std::vector<float>& a,
                              const std::vector<float>& b, Shape shape) {
  const auto [m, n, k] = shape;
  std::vector<double> out(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int x = 0; x < k; ++x) {
        sum += static_cast<double>(a[row * k + x]) * b[x * n + col];
      }
      out[row * n + col] = sum;
    }
  }
  return out;
}

void check(const std::vector<float>& got, const std::vector<double>& expected,
           Shape shape, const std::string& method) {
  const auto [m, n, k] = shape;
  for (size_t i = 0; i < got.size(); ++i) {
    const double tolerance = 5e-4 + 1e-4 * std::abs(expected[i]);
    if (!std::isfinite(got[i]) || std::abs(got[i] - expected[i]) > tolerance) {
      throw std::runtime_error(method + " mismatch at (" +
          std::to_string(i / n) + "," + std::to_string(i % n) + ") for " +
          std::to_string(m) + "x" + std::to_string(n) + "x" +
          std::to_string(k) + ": got " + std::to_string(got[i]) +
          ", expected " + std::to_string(expected[i]));
    }
  }
}

void run_case(Shape shape, std::mt19937& rng, cublasHandle_t handle) {
  const auto [m, n, k] = shape;
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> a(static_cast<size_t>(m) * k);
  std::vector<float> b(static_cast<size_t>(k) * n);
  for (float& value : a) value = distribution(rng);
  for (float& value : b) value = distribution(rng);
  const auto expected = reference(a, b, shape);
  std::vector<float> got(static_cast<size_t>(m) * n);
  matmul_cpu(a.data(), b.data(), got.data(), m, n, k);
  check(got, expected, shape, "CPU");

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, got.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

  const std::vector<float> poison(got.size(), std::numeric_limits<float>::quiet_NaN());
  for (const auto& [name, kernel] :
       std::vector<std::pair<std::string, Kernel>>{{"naive", launch_matmul_naive},
                                                    {"tiled", launch_matmul_tiled}}) {
    CUDA_CHECK(cudaMemcpy(dc, poison.data(), poison.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    kernel(da, db, dc, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), dc, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    check(got, expected, shape, name);
  }
  CUDA_CHECK(cudaMemcpy(dc, poison.data(), poison.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  matmul_cublas(handle, da, db, dc, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(got.data(), dc, got.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  check(got, expected, shape, "cuBLAS FP32 pedantic");
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
}
}  // namespace

int main() {
  try {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("No CUDA device found");
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasCreate failed");
    }
    std::mt19937 rng(20261007);
    std::vector<Shape> shapes = {
        {1, 1, 1}, {1, 17, 19}, {15, 16, 17}, {16, 16, 16},
        {17, 31, 33}, {31, 17, 15}, {65, 7, 47}, {7, 65, 47},
        {127, 129, 63}, {33, 64, 257}};
    std::uniform_int_distribution<int> dimension(1, 73);
    for (int i = 0; i < 20; ++i) {
      shapes.emplace_back(dimension(rng), dimension(rng), dimension(rng));
    }
    for (const Shape& shape : shapes) run_case(shape, rng, handle);
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasDestroy failed");
    }
    std::cout << "GEMM correctness passed: " << shapes.size()
              << " deterministic shapes x 4 implementations\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

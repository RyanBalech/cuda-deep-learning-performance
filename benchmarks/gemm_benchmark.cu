#include "cpu_baselines.h"
#include "cublas_baseline.cuh"
#include "cuda_check.cuh"
#include "matmul.cuh"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Shape = std::tuple<int, int, int, int, bool>;  // M, N, K, GPU iterations, CPU timing.

template <typename Launch>
void time_gpu_batch(std::ostream& out, const char* name, Launch launch,
                    int m, int n, int k, int rep, int iterations) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  out << name << ',' << m << ',' << n << ',' << k << ',' << rep << ','
      << 10 << ',' << iterations << ',' << elapsed_ms << ','
      << elapsed_ms / iterations << ",cuda_event\n";
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void time_cpu(std::ostream& out, const float* a, const float* b, float* c,
              int m, int n, int k) {
  constexpr int repetitions = 7;
  matmul_cpu(a, b, c, m, n, k);
  for (int rep = 0; rep < repetitions; ++rep) {
    const auto start = std::chrono::steady_clock::now();
    matmul_cpu(a, b, c, m, n, k);
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    out << "cpu," << m << ',' << n << ',' << k << ',' << rep
        << ",1,1," << ms << ',' << ms << ",steady_clock\n";
  }
}

void run_shape(std::ostream& out, Shape shape, std::mt19937& rng,
               cublasHandle_t handle) {
  const int m = std::get<0>(shape);
  const int n = std::get<1>(shape);
  const int k = std::get<2>(shape);
  const int iterations = std::get<3>(shape);
  const bool include_cpu = std::get<4>(shape);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> a(static_cast<size_t>(m) * k);
  std::vector<float> b(static_cast<size_t>(k) * n);
  std::vector<float> c(static_cast<size_t>(m) * n);
  for (float& value : a) value = distribution(rng);
  for (float& value : b) value = distribution(rng);
  if (include_cpu) time_cpu(out, a.data(), b.data(), c.data(), m, n, k);

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (int i = 0; i < 10; ++i) {
    launch_matmul_naive(da, db, dc, m, n, k);
    launch_matmul_tiled(da, db, dc, m, n, k);
    matmul_cublas(handle, da, db, dc, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  // Rotate method order to reduce systematic clock and temperature bias.
  for (int rep = 0; rep < 15; ++rep) {
    for (int offset = 0; offset < 3; ++offset) {
      switch ((rep + offset) % 3) {
        case 0:
          time_gpu_batch(out, "naive", [&] {
            launch_matmul_naive(da, db, dc, m, n, k);
          }, m, n, k, rep, iterations);
          break;
        case 1:
          time_gpu_batch(out, "tiled", [&] {
            launch_matmul_tiled(da, db, dc, m, n, k);
          }, m, n, k, rep, iterations);
          break;
        case 2:
          time_gpu_batch(out, "cublas_fp32_pedantic", [&] {
            matmul_cublas(handle, da, db, dc, m, n, k);
          }, m, n, k, rep, iterations);
          break;
      }
    }
  }
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--output") {
      throw std::runtime_error("Usage: gemm_benchmark --output path/to/raw.csv");
    }
    std::ofstream out(argv[2]);
    if (!out) throw std::runtime_error("Could not open output CSV");
    out << "method,m,n,k,repetition,warmups,iterations,batch_ms,per_call_ms,timing\n";
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasCreate failed");
    }
    std::mt19937 rng(20261007);
    for (const Shape& shape : std::vector<Shape>{{128, 128, 128, 50, true},
                                                  {255, 257, 129, 20, true},
                                                  {512, 512, 512, 10, false},
                                                  {1024, 1024, 1024, 3, false}}) {
      run_shape(out, shape, rng, handle);
      out.flush();
    }
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasDestroy failed");
    }
    std::cout << "Wrote raw benchmark measurements to " << argv[2] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

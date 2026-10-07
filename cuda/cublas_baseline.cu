#include "cublas_baseline.cuh"

#include <stdexcept>
#include <string>

void matmul_cublas(cublasHandle_t handle, const float* a, const float* b,
                   float* c, int m, int n, int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major C[M,N] is column-major C^T[N,M] = B^T[N,K] * A^T[K,M].
  const cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
      b, CUDA_R_32F, n, a, CUDA_R_32F, k, &beta,
      c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasGemmEx failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

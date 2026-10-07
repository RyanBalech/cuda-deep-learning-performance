#include "matmul.cuh"
#include "cuda_check.cuh"

namespace {
constexpr int kTile = 16;

__global__ void naive(const float* a, const float* b, float* c,
                      int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float sum = 0.0f;
  for (int x = 0; x < k; ++x) {
    sum += a[row * k + x] * b[x * n + col];
  }
  c[row * n + col] = sum;
}

__global__ void tiled(const float* a, const float* b, float* c,
                      int m, int n, int k) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int x = 0; x < kTile; ++x) {
      sum += a_tile[threadIdx.y][x] * b_tile[x][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}
}  // namespace

void launch_matmul_naive(const float* a, const float* b, float* c,
                         int m, int n, int k) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  naive<<<grid, block>>>(a, b, c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_tiled(const float* a, const float* b, float* c,
                         int m, int n, int k) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  tiled<<<grid, block>>>(a, b, c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

#pragma once

// Row-major C[M,N] = A[M,K] * B[K,N], all buffers on the GPU.
void launch_matmul_naive(const float* a, const float* b, float* c,
                         int m, int n, int k);
void launch_matmul_tiled(const float* a, const float* b, float* c,
                         int m, int n, int k);

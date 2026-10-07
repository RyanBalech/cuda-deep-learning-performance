#pragma once

// Row-major C[M,N] = A[M,K] * B[K,N], all inputs FP32.
void matmul_cpu(const float* a, const float* b, float* c, int m, int n, int k);

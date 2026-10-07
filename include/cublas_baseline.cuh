#pragma once

#include <cublas_v2.h>

// FP32 inputs, FP32 accumulation, no TF32. The handle is reused across calls.
void matmul_cublas(cublasHandle_t handle, const float* a, const float* b,
                   float* c, int m, int n, int k);

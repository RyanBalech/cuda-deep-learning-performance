#include "cpu_baselines.h"

void matmul_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int x = 0; x < k; ++x) {
        sum += a[row * k + x] * b[x * n + col];
      }
      c[row * n + col] = sum;
    }
  }
}

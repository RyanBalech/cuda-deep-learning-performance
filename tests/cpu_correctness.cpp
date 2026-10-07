#include "cpu_baselines.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::mt19937 rng(20261007);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f);
  for (int trial = 0; trial < 30; ++trial) {
    const int m = 1 + (trial * 7) % 33;
    const int n = 1 + (trial * 11) % 35;
    const int k = 1 + (trial * 13) % 37;
    std::vector<float> a(m * k), b(k * n), c(m * n, NAN);
    for (float& x : a) x = values(rng);
    for (float& x : b) x = values(rng);
    matmul_cpu(a.data(), b.data(), c.data(), m, n, k);
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        double reference = 0;
        for (int inner = 0; inner < k; ++inner)
          reference += static_cast<double>(a[row * k + inner]) * b[inner * n + col];
        const double error = std::abs(c[row * n + col] - reference);
        if (!std::isfinite(c[row * n + col]) || error > 5e-4 + 1e-4 * std::abs(reference)) {
          std::cerr << "CPU reference mismatch at trial " << trial << '\n';
          return 1;
        }
      }
    }
  }
  std::cout << "CPU baseline: 30 rectangular cases passed\n";
}

# CUDA Deep Learning Performance Lab

A systems-oriented study of how CUDA execution, memory hierarchy, tiling, and kernel fusion affect core deep-learning operators.

## Goal
Build correct CPU/GPU implementations first, then optimize them under controlled benchmarks. The focus is GPU developer technology: identify bottlenecks, measure them, and explain why an optimization helps.

## Study
- GEMM: CPU, naive CUDA, shared-memory tiled CUDA, PyTorch/cuBLAS comparison
- Reduction and softmax: parallel reduction, synchronization, numerical stability
- Fused Linear + ReLU: kernel fusion and memory traffic
- Attention: tiling, fusion, arithmetic intensity
- PyTorch integration: custom C++/CUDA extension

Every optimized kernel must pass numerical correctness tests before performance is reported. No speedup is claimed until measured on actual NVIDIA hardware.

## Build
Requires NVIDIA CUDA toolkit and CMake >= 3.24.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See `docs/PERFORMANCE_STUDY.md` for the experimental protocol.

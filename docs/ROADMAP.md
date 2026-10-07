# NVIDIA DevTech-aligned roadmap

## Implemented foundation
- C++17/CUDA build
- naive and shared-memory tiled GEMM
- parallel reduction
- numerically stable row softmax
- fused bias + ReLU
- custom PyTorch C++/CUDA extension
- correctness tests and CUDA-event benchmarks
- transparent attention reference kernel
- reproducible hardware metadata and profiling protocol

## Next experimental milestones
1. Run correctness suite on NVIDIA hardware.
2. Collect GEMM scaling data and compare naive vs tiled kernels.
3. Add cuBLAS/PyTorch library baselines.
4. Profile global-memory transactions, occupancy and achieved throughput with Nsight Compute.
5. Benchmark fused vs unfused bias/ReLU.
6. Optimize reduction with warp shuffles and compare against the shared-memory baseline.
7. Replace the transparent attention reference with a tiled/fused variant and quantify where it helps.
8. Write conclusions from measured evidence, including optimizations that fail to improve performance.

The project intentionally separates implementation from empirical claims. A CV speedup number is added only after a reproducible measurement.

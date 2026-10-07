# CUDA GEMM performance study

This project compares four row-major, FP32 matrix-multiplication implementations: a scalar CPU reference, a direct CUDA kernel, a 16×16 shared-memory tiled CUDA kernel, and cuBLAS. The current scope is **GEMM only**. It does not implement softmax, attention, fusion, or a PyTorch extension.

For `A[M,K]` and `B[K,N]`, every implementation produces `C[M,N]`. The cuBLAS path uses `CUBLAS_COMPUTE_32F_PEDANTIC` so its arithmetic is FP32 rather than TF32. The CPU path also accumulates in FP32; correctness is checked against a separate FP64 reference.

## Build and verify

Prerequisites: an NVIDIA GPU and driver, CUDA Toolkit, CMake 3.24 or newer, and a CUDA-supported C++ compiler. On Windows, the tested configuration uses Microsoft C++ Build Tools 2026 and the x64 release build. If the toolkit was installed without adding paths to the shell, set them before configuring and running:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_PATH_V13_2 = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The correctness executable tests the CPU, direct CUDA, tiled CUDA, and cuBLAS implementations on 30 deterministic cases. These include random inputs, rectangular shapes, and dimensions below, exactly on, and above the 16-element tile boundary. Each result is compared against an FP64 reference with a fixed absolute and relative tolerance. Output buffers are filled with NaNs before GPU calls to detect unwritten elements.

The verified release build passed all 30 cases. Compute Sanitizer memcheck, racecheck, synccheck, and initcheck reported zero errors. Nsight Compute profiles were collected for both handwritten kernels at 512³. The [performance study](docs/PERFORMANCE_STUDY.md) records the exact commands, measurements, hardware, and limits.

## Benchmark

Run the correctness test before timing. The benchmark uses nonzero seeded inputs and records every repetition in CSV:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\build\Release\gemm_benchmark.exe --output results\raw_gemm.csv
python scripts\summarize.py results\raw_gemm.csv results\summary_gemm.csv
```

GPU time is measured with CUDA events on the default stream, after 10 warmups, across 15 independent batches per method and shape. Each batch contains multiple calls; the raw CSV reports batch time and per-call time. CPU time uses `steady_clock`, one warmup and seven repetitions, and is reported separately. GPU timings exclude allocation and host↔device transfers; CPU timing covers only multiplication. The benchmark includes `128³`, `512³`, `1024³`, and a rectangular `255×257×129` case. The CPU baseline is timed on the two smaller cases. The summary script reports median, 10th and 90th percentile per-call latency plus throughput calculated as `2MNK / time`.

On the RTX 4050 Laptop GPU, median tiled-kernel latency was **0.423 ms at 512³** and **3.193 ms at 1024³**, about **1.30× faster** than the direct kernel at both sizes. Strict-FP32 cuBLAS measured **0.0878 ms** and **0.495 ms** respectively. These are kernel-only, device-specific results; the small-shape measurements have wider relative variation. See [performance study](docs/PERFORMANCE_STUDY.md) and the retained files under [`results/`](results) for the full evidence.

# GEMM performance study

## Scope and method

This study compares row-major FP32 GEMM on the local RTX 4050 Laptop GPU. It covers the existing direct kernel and 16×16 shared-memory tiled kernel, the connected scalar CPU baseline, and a cuBLAS baseline set to `CUBLAS_COMPUTE_32F_PEDANTIC`. The input matrices contain seeded, nonzero random values in [-1, 1]. GPU timings exclude allocation and transfers; CPU timings cover only multiplication.

Correctness was checked first on 30 deterministic shapes, including 20 seeded random shapes, rectangles, and dimensions immediately below, on, and above a 16-element tile boundary. Each CPU and GPU output was compared elementwise with an FP64 reference using `|observed - expected| ≤ 5e-4 + 1e-4 × |expected|`. Poisoning the GPU output buffers with NaNs before each call also checks that edge tiles write every valid output. The release build passed CTest; its console output is saved in [correctness log](../results/correctness_2026-10-07.txt).

GPU measurements used CUDA events, 10 warmups, and 15 measured batches per method and shape. Multiple launches per batch amortize event overhead, and method order rotates across repetitions. CPU measurements used `steady_clock`, one warmup, and seven repetitions on the two smaller shapes. The [raw CSV](../results/raw_gemm_2026-10-07.csv) retains every batch; the [summary CSV](../results/summary_gemm_2026-10-07.csv) gives medians and 10th–90th percentile ranges.

## Measured latency

Median milliseconds per multiplication, on the hardware described in [environment](../results/environment_2026-10-07.md):

| M×N×K | CPU | Direct CUDA | Tiled CUDA | cuBLAS FP32 pedantic |
| --- | ---: | ---: | ---: | ---: |
| 128×128×128 | 1.3507 | 0.0168 | 0.0141 | 0.0160 |
| 255×257×129 | 4.0466 | 0.0418 | 0.0370 | 0.0203 |
| 512×512×512 | — | 0.5513 | 0.4228 | 0.0878 |
| 1024×1024×1024 | — | 4.1622 | 3.1932 | 0.4949 |

At 512³ and 1024³, tiling reduced median latency by about **1.30×** versus the direct kernel. Strict-FP32 cuBLAS was **4.82×** and **6.45×** faster than the tiled kernel on those shapes. The 128³ medians are close and the raw repetitions show much wider relative variation, so that shape does not support a strong ranking. CPU results should not be read as end-to-end GPU speedups: transfers are excluded and the clocks measure different execution paths.

## Sanitizer and profiler

Compute Sanitizer ran the correctness executable under [memcheck](../results/compute_sanitizer_memcheck_2026-10-07.txt), [racecheck](../results/compute_sanitizer_racecheck_2026-10-07.txt), [synccheck](../results/compute_sanitizer_synccheck_2026-10-07.txt), and [initcheck](../results/compute_sanitizer_initcheck_2026-10-07.txt). Each completed with zero reported errors; racecheck reported zero hazards and warnings. Windows WDDM required enabling NVIDIA's documented GPU debugger interface before these runs.

Nsight Compute's basic set profiled one 512³ launch of each handwritten kernel. The [direct-kernel report](../results/nsight_compute_naive_metrics_2026-10-07.txt) recorded 545.12 µs, 97.83% L1/TEX cache throughput, and 2.01% DRAM throughput. The [tiled-kernel report](../results/nsight_compute_metrics_2026-10-07.txt) recorded 416.96 µs, 96.17% L1/TEX cache throughput, and 2.63% DRAM throughput. Both showed high load/store pipe activity. This suggests on-chip memory instruction pressure is relevant at 512³, but one basic profile per kernel does not isolate a single bottleneck. The native `.ncu-rep` files are retained beside the text exports in `results/`.

These are laptop GPU results under WDDM and a 30 W power limit. They are not hardware-independent performance claims. Short kernels also include possible host launch gaps between calls within CUDA-event batches. No transfer-inclusive, mixed-precision, tensor-core, softmax, attention, or fusion result is claimed.

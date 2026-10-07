# Profiling

Use Nsight Compute for kernel-level analysis and Nsight Systems for end-to-end timelines.

Suggested questions:
- Are global loads coalesced?
- Does shared-memory tiling increase data reuse?
- Is the kernel limited by memory throughput, instruction throughput, occupancy, or launch overhead?
- Does fusion remove a measurable intermediate memory round trip?
- How does the bottleneck change with tensor shape?

Profiler reports themselves are ignored because they are machine-specific and large. Record derived metrics in the performance study with hardware metadata.

cmake_minimum_required(VERSION 3.24)
project(cuda_dl_performance LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include(CTest)

add_library(cuda_kernels
  cuda/matmul.cu
  cuda/reduction.cu
  cuda/softmax.cu
  cuda/fused_ops.cu
  cuda/attention.cu)
target_include_directories(cuda_kernels PUBLIC include)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gemm_benchmark benchmarks/gemm_benchmark.cu)
target_link_libraries(gemm_benchmark PRIVATE cuda_kernels)
add_executable(ops_benchmark benchmarks/ops_benchmark.cu)
target_link_libraries(ops_benchmark PRIVATE cuda_kernels)

if(BUILD_TESTING)
  add_executable(gemm_correctness tests/gemm_correctness.cu)
  target_link_libraries(gemm_correctness PRIVATE cuda_kernels)
  add_test(NAME gemm_correctness COMMAND gemm_correctness)
  add_executable(ops_correctness tests/ops_correctness.cu)
  target_link_libraries(ops_correctness PRIVATE cuda_kernels)
  add_test(NAME ops_correctness COMMAND ops_correctness)
endif()

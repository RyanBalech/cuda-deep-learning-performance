#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
inline void cuda_check(cudaError_t s,const char* e){if(s!=cudaSuccess)throw std::runtime_error(std::string(e)+": "+cudaGetErrorString(s));}
#define CUDA_CHECK(expr) cuda_check((expr),#expr)

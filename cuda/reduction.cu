#include "ops.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
namespace {
__global__ void reduce_sum_kernel(const float* x, float* partial, int n) {
    extern __shared__ float s[];
    unsigned tid=threadIdx.x;
    unsigned i=blockIdx.x*(blockDim.x*2)+tid;
    float v=0.0f;
    if(i<n) v+=x[i];
    if(i+blockDim.x<n) v+=x[i+blockDim.x];
    s[tid]=v; __syncthreads();
    for(unsigned stride=blockDim.x/2; stride>0; stride>>=1) {
        if(tid<stride) s[tid]+=s[tid+stride];
        __syncthreads();
    }
    if(tid==0) partial[blockIdx.x]=s[0];
}
}
void launch_reduce_sum(const float* x,float* out,int n) {
    constexpr int block=256;
    int blocks=(n+block*2-1)/(block*2);
    float* partial=nullptr;
    CUDA_CHECK(cudaMalloc(&partial, blocks*sizeof(float)));
    reduce_sum_kernel<<<blocks,block,block*sizeof(float)>>>(x,partial,n);
    CUDA_CHECK(cudaGetLastError());
    if(blocks==1) CUDA_CHECK(cudaMemcpyAsync(out,partial,sizeof(float),cudaMemcpyDeviceToDevice));
    else launch_reduce_sum(partial,out,blocks);
    CUDA_CHECK(cudaFree(partial));
}

#include "ops.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
namespace {
__global__ void bias_relu_kernel(const float*x,const float*b,float*y,int rows,int cols){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;
 if(i<n){float v=x[i]+b[i%cols];y[i]=v>0.0f?v:0.0f;}
}}
void launch_bias_relu(const float*x,const float*b,float*y,int rows,int cols){
 int n=rows*cols,block=256;
 bias_relu_kernel<<<(n+block-1)/block,block>>>(x,b,y,rows,cols);
 CUDA_CHECK(cudaGetLastError());
}

#include "ops.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
#include <cfloat>
namespace {
__global__ void softmax_rows(const float* x,float* y,int rows,int cols) {
    extern __shared__ float s[];
    int row=blockIdx.x, tid=threadIdx.x;
    if(row>=rows) return;
    float local=-FLT_MAX;
    for(int c=tid;c<cols;c+=blockDim.x) local=fmaxf(local,x[row*cols+c]);
    s[tid]=local; __syncthreads();
    for(int d=blockDim.x/2;d>0;d>>=1){if(tid<d)s[tid]=fmaxf(s[tid],s[tid+d]);__syncthreads();}
    float maxv=s[0],sum=0.0f;
    for(int c=tid;c<cols;c+=blockDim.x) sum+=expf(x[row*cols+c]-maxv);
    s[tid]=sum; __syncthreads();
    for(int d=blockDim.x/2;d>0;d>>=1){if(tid<d)s[tid]+=s[tid+d];__syncthreads();}
    float inv=1.0f/s[0];
    for(int c=tid;c<cols;c+=blockDim.x)y[row*cols+c]=expf(x[row*cols+c]-maxv)*inv;
}
}
void launch_softmax(const float* x,float* y,int rows,int cols){
    constexpr int block=256;
    softmax_rows<<<rows,block,block*sizeof(float)>>>(x,y,rows,cols);
    CUDA_CHECK(cudaGetLastError());
}

#include "attention.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
// Educational reference kernel: one block computes one query row.
// It deliberately prioritizes transparent dataflow over production performance.
namespace {
__global__ void attention_kernel(const float*q,const float*k,const float*v,float*out,int seq,int dim){
 extern __shared__ float sm[]; float* scores=sm;
 int row=blockIdx.x,tid=threadIdx.x;
 if(row>=seq)return;
 if(tid==0){
  float mx=-FLT_MAX,scale=rsqrtf((float)dim);
  for(int j=0;j<seq;++j){float dot=0;for(int d=0;d<dim;++d)dot+=q[row*dim+d]*k[j*dim+d];scores[j]=dot*scale;mx=fmaxf(mx,scores[j]);}
  float denom=0;for(int j=0;j<seq;++j){scores[j]=expf(scores[j]-mx);denom+=scores[j];}
  for(int j=0;j<seq;++j)scores[j]/=denom;
 }
 __syncthreads();
 for(int d=tid;d<dim;d+=blockDim.x){float acc=0;for(int j=0;j<seq;++j)acc+=scores[j]*v[j*dim+d];out[row*dim+d]=acc;}
}}
void launch_attention_naive(const float*q,const float*k,const float*v,float*out,int seq,int dim){
 attention_kernel<<<seq,256,seq*sizeof(float)>>>(q,k,v,out,seq,dim);CUDA_CHECK(cudaGetLastError());
}

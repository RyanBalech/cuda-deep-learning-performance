#include "cuda_check.cuh"
#include "matmul.cuh"
#include <iostream>
#include <vector>
using Launcher=void(*)(const float*,const float*,float*,int,int,int);
float bench(Launcher f,const float*a,const float*b,float*c,int s,int trials=100){
 for(int i=0;i<10;++i)f(a,b,c,s,s,s);CUDA_CHECK(cudaDeviceSynchronize());
 cudaEvent_t x,y;CUDA_CHECK(cudaEventCreate(&x));CUDA_CHECK(cudaEventCreate(&y));CUDA_CHECK(cudaEventRecord(x));
 for(int i=0;i<trials;++i)f(a,b,c,s,s,s);CUDA_CHECK(cudaEventRecord(y));CUDA_CHECK(cudaEventSynchronize(y));float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,x,y));CUDA_CHECK(cudaEventDestroy(x));CUDA_CHECK(cudaEventDestroy(y));return ms/trials;
}
int main(){std::cout<<"size,naive_ms,tiled_ms,naive_gflops,tiled_gflops\n";for(int s:{256,512,1024,2048}){size_t z=size_t(s)*s;float *a,*b,*c;CUDA_CHECK(cudaMalloc(&a,z*4));CUDA_CHECK(cudaMalloc(&b,z*4));CUDA_CHECK(cudaMalloc(&c,z*4));CUDA_CHECK(cudaMemset(a,0,z*4));CUDA_CHECK(cudaMemset(b,0,z*4));float n=bench(launch_matmul_naive,a,b,c,s),t=bench(launch_matmul_tiled,a,b,c,s);double ops=2.0*s*s*s;std::cout<<s<<","<<n<<","<<t<<","<<ops/(n*1e6)<<","<<ops/(t*1e6)<<"\n";CUDA_CHECK(cudaFree(a));CUDA_CHECK(cudaFree(b));CUDA_CHECK(cudaFree(c));}}

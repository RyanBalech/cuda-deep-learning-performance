#include "cuda_check.cuh"
#include "ops.cuh"
#include <iostream>
template<class F> float time_ms(F f,int warmup=10,int trials=100){
 for(int i=0;i<warmup;++i)f();CUDA_CHECK(cudaDeviceSynchronize());
 cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
 for(int i=0;i<trials;++i)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms/trials;
}
int main(){
 std::cout<<"operator,shape,ms\n";
 for(int cols:{128,512,2048,8192}){int rows=1024;float *x,*y;cudaMalloc(&x,size_t(rows)*cols*4);cudaMalloc(&y,size_t(rows)*cols*4);float ms=time_ms([&]{launch_softmax(x,y,rows,cols);});std::cout<<"softmax,"<<rows<<"x"<<cols<<","<<ms<<"\n";cudaFree(x);cudaFree(y);}
}

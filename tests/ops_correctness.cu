#include "cuda_check.cuh"
#include "ops.cuh"
#include <cmath>
#include <iostream>
#include <vector>
bool close(float a,float b,float eps=1e-4f){return std::fabs(a-b)<eps;}
int main(){
 {int n=1003;std::vector<float>h(n);float ref=0;for(int i=0;i<n;++i){h[i]=(i%13-6)*0.1f;ref+=h[i];}
 float *x,*o;CUDA_CHECK(cudaMalloc(&x,n*4));CUDA_CHECK(cudaMalloc(&o,4));CUDA_CHECK(cudaMemcpy(x,h.data(),n*4,cudaMemcpyHostToDevice));launch_reduce_sum(x,o,n);float got;CUDA_CHECK(cudaMemcpy(&got,o,4,cudaMemcpyDeviceToHost));if(!close(got,ref,2e-3f))return 1;cudaFree(x);cudaFree(o);}
 {int r=3,c=17;std::vector<float>h(r*c);for(int i=0;i<r*c;++i)h[i]=(i%11-5)*0.2f;float *x,*y;CUDA_CHECK(cudaMalloc(&x,h.size()*4));CUDA_CHECK(cudaMalloc(&y,h.size()*4));CUDA_CHECK(cudaMemcpy(x,h.data(),h.size()*4,cudaMemcpyHostToDevice));launch_softmax(x,y,r,c);std::vector<float>g(h.size());CUDA_CHECK(cudaMemcpy(g.data(),y,g.size()*4,cudaMemcpyDeviceToHost));for(int row=0;row<r;++row){float s=0;for(int j=0;j<c;++j)s+=g[row*c+j];if(!close(s,1.0f,1e-4f))return 2;}cudaFree(x);cudaFree(y);}
 std::cout<<"Operator correctness passed\n";
}

#include "cuda_check.cuh"
#include "matmul.cuh"
#include <cmath>
#include <iostream>
#include <vector>
int main(){
 constexpr int m=7,n=5,k=9; std::vector<float>a(m*k),b(k*n),ref(m*n,0);
 for(int i=0;i<m*k;++i)a[i]=float((i%7)-3)/7;
 for(int i=0;i<k*n;++i)b[i]=float((i%5)-2)/5;
 for(int r=0;r<m;++r)for(int c=0;c<n;++c)for(int x=0;x<k;++x)ref[r*n+c]+=a[r*k+x]*b[x*n+c];
 float *da,*db,*dc; CUDA_CHECK(cudaMalloc(&da,a.size()*4));CUDA_CHECK(cudaMalloc(&db,b.size()*4));CUDA_CHECK(cudaMalloc(&dc,ref.size()*4));
 CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
 for(auto launch:{launch_matmul_naive,launch_matmul_tiled}){launch(da,db,dc,m,n,k);CUDA_CHECK(cudaDeviceSynchronize());std::vector<float>got(m*n);CUDA_CHECK(cudaMemcpy(got.data(),dc,got.size()*4,cudaMemcpyDeviceToHost));for(size_t i=0;i<got.size();++i)if(std::fabs(got[i]-ref[i])>1e-4){std::cerr<<"Mismatch "<<i<<"\n";return 1;}}
 CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));std::cout<<"GEMM correctness passed\n";
}

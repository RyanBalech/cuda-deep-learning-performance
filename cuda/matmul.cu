#include "matmul.cuh"
#include "cuda_check.cuh"
namespace { constexpr int TILE=16;
__global__ void naive(const float*a,const float*b,float*c,int m,int n,int k){
 int r=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
 if(r>=m||col>=n)return; float sum=0;
 for(int x=0;x<k;++x)sum+=a[r*k+x]*b[x*n+col]; c[r*n+col]=sum;
}
__global__ void tiled(const float*a,const float*b,float*c,int m,int n,int k){
 __shared__ float as[TILE][TILE],bs[TILE][TILE];
 int r=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x; float sum=0;
 for(int t=0;t<(k+TILE-1)/TILE;++t){
  int ac=t*TILE+threadIdx.x,br=t*TILE+threadIdx.y;
  as[threadIdx.y][threadIdx.x]=(r<m&&ac<k)?a[r*k+ac]:0;
  bs[threadIdx.y][threadIdx.x]=(br<k&&col<n)?b[br*n+col]:0;
  __syncthreads();
  #pragma unroll
  for(int x=0;x<TILE;++x)sum+=as[threadIdx.y][x]*bs[x][threadIdx.x];
  __syncthreads();
 }
 if(r<m&&col<n)c[r*n+col]=sum;
}}
void launch_matmul_naive(const float*a,const float*b,float*c,int m,int n,int k){dim3 block(TILE,TILE),grid((n+TILE-1)/TILE,(m+TILE-1)/TILE);naive<<<grid,block>>>(a,b,c,m,n,k);CUDA_CHECK(cudaGetLastError());}
void launch_matmul_tiled(const float*a,const float*b,float*c,int m,int n,int k){dim3 block(TILE,TILE),grid((n+TILE-1)/TILE,(m+TILE-1)/TILE);tiled<<<grid,block>>>(a,b,c,m,n,k);CUDA_CHECK(cudaGetLastError());}

#include <torch/extension.h>
#include "ops.cuh"
torch::Tensor softmax_cuda(torch::Tensor x){
 TORCH_CHECK(x.is_cuda(),"x must be CUDA");TORCH_CHECK(x.scalar_type()==torch::kFloat32,"float32 only");TORCH_CHECK(x.dim()==2,"expected 2D tensor");
 auto y=torch::empty_like(x);launch_softmax(x.data_ptr<float>(),y.data_ptr<float>(),x.size(0),x.size(1));return y;
}
torch::Tensor bias_relu_cuda(torch::Tensor x,torch::Tensor bias){
 TORCH_CHECK(x.is_cuda()&&bias.is_cuda(),"tensors must be CUDA");TORCH_CHECK(x.scalar_type()==torch::kFloat32&&bias.scalar_type()==torch::kFloat32,"float32 only");TORCH_CHECK(x.dim()==2&&bias.dim()==1&&bias.size(0)==x.size(1),"shape mismatch");
 auto y=torch::empty_like(x);launch_bias_relu(x.data_ptr<float>(),bias.data_ptr<float>(),y.data_ptr<float>(),x.size(0),x.size(1));return y;
}

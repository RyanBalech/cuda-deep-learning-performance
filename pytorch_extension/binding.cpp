#include <torch/extension.h>
torch::Tensor softmax_cuda(torch::Tensor x);
torch::Tensor bias_relu_cuda(torch::Tensor x,torch::Tensor bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("softmax",&softmax_cuda,"Row softmax (CUDA)");
 m.def("bias_relu",&bias_relu_cuda,"Fused bias + ReLU (CUDA)");
}

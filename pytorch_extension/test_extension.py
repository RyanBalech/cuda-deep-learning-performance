import torch
import cuda_dl_ops
def check():
    torch.manual_seed(0)
    x=torch.randn(128,513,device="cuda",dtype=torch.float32)
    y=cuda_dl_ops.softmax(x)
    torch.testing.assert_close(y,torch.softmax(x,dim=-1),rtol=1e-4,atol=1e-5)
    b=torch.randn(513,device="cuda")
    z=cuda_dl_ops.bias_relu(x,b)
    torch.testing.assert_close(z,torch.relu(x+b),rtol=1e-5,atol=1e-6)
    print("PyTorch extension correctness passed")
if __name__=="__main__": check()

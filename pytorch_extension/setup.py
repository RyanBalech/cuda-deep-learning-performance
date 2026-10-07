from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="cuda_dl_ops",ext_modules=[CUDAExtension(name="cuda_dl_ops",sources=["binding.cpp","extension.cu","../cuda/softmax.cu","../cuda/fused_ops.cu"],include_dirs=["../include"])],cmdclass={"build_ext":BuildExtension})

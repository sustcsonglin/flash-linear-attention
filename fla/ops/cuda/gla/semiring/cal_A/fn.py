import os
from logging import warning

from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

try:
    cuda_cal_A = load(
        name="cuda_cal_A",
        sources=[os.path.join(module_path, "inner_chunk16_dim16x.cpp"), os.path.join(
            module_path, "inner_chunk16_dim16x.cu")],
        # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
        verbose=False,
    )
    cuda_cal_A_bf16 = load(
        name="cuda_cal_A_bf16",
        sources=[os.path.join(module_path, "inner_chunk16_dim16x_bf16.cpp"), os.path.join(
            module_path, "inner_chunk16_dim16x_bf16.cu")],
        # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
        verbose=False,
    )
except ImportError:
    cuda_cal_A = None
    cuda_cal_A_bf16 = None
    warning('Failed to import cuda_cal_A, cuda_cal_A_bf16 which are needed for FusedChunk computation in GLA')

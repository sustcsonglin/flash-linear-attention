import os
from logging import warning

from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

try:
    semiring_cal_A = load(
        name="semiring_cal_A",
        sources=[os.path.join(module_path, "inner_chunk16_dim16x.cpp"), os.path.join(
            module_path, "inner_chunk16_dim16x.cu")],
        # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
        verbose=True,
    )
except ImportError:
    semiring_cal_A = None
    warning('Failed to import semiring_cal_A. Do not use FusedChunk implementation of GLA.')


if "__main__" == __name__:
    import time

    import torch
    from einops import rearrange

    batch = 1
    num_head = 1
    seq_len = 4

    d_head = 16
    dtype = torch.bfloat16
    q = torch.randn(batch, num_head, seq_len, 16, d_head,
                    dtype=dtype, requires_grad=True).cuda()
    k = torch.randn(batch, num_head, seq_len, 16, d_head,
                    dtype=dtype, requires_grad=True).cuda()
    g = torch.randn(batch, num_head, seq_len, 16, d_head,
                    dtype=dtype, requires_grad=True).cuda()
    o = semiring_cal_A.forward(q, k, g)
    do = torch.randn_like(o)
    dq, dk, dg = semiring_cal_A.backward(q, k, g, do)
    breakpoint()

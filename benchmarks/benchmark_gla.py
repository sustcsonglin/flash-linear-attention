# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import math
import pickle
from tabnanny import verbose

import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark import (benchmark_all, benchmark_backward, benchmark_combined,
                       benchmark_forward, benchmark_fwd_bwd)
from einops import rearrange, repeat
from fla.ops.triton.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

# try:
#     from triton.ops.flash_attention import attention as attention_triton
# except ImportError:
#     attention_triton = None

# try:
#     import xformers.ops as xops
# except ImportError:
#     xops = None


def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_combined(func, *args, **kwargs)
    return time_fb[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [True]
headdim_vals = [64, 128, 256, 512]
dim = 2048
dropout_p = 0.0

methods = (["fused_chunk", "fused_recurrent", "chunk"])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim

            q = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            g = F.logsigmoid(torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True)).clamp_min(-5).requires_grad_(True)
            v = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            
            fb = time_fwd_bwd(
               fused_chunk_gla, q, k, v, g, verbose=False
            )
            time_f_b[config, "fused_chunk"] = fb
            # time_b[config, "fused_chunk"] = b
            
            q2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            g2 = F.logsigmoid(torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)).clamp_min(-5).requires_grad_(True)
            v2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            f_b = time_fwd_bwd(
               fused_recurrent_gla, q2, k2, v2, g2,  verbose=False
            )
            time_f_b[config, "fused_recurrent"] = f_b 

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            q2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            g2 = F.logsigmoid(torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)).clamp_min(-5).requires_grad_(True)
            v2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            f_b = time_fwd_bwd(
               chunk_gla, q2, k2, v2, g2,  verbose=False
            )
            time_f_b[config, "chunk"] = f_b 

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(
                    f"{method} fwd + bwd: {time_f_b[config, method]:.10f} "
                )

                # speed_f[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                #     time_f[config, method]
                # )
                # speed_b[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                #     time_b[config, method]
                # )
                # speed_f_b[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                #     time_f_b[config, method]
                # )
                # print(
                #     f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                #     f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                #     f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                # )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
                

                
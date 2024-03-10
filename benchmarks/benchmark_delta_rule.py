# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"

import torch
from benchmark import benchmark_combined, benchmark_forward

from fla.ops.retention import chunk_retention, fused_chunk_retention
from fla.ops.delta_rule import fused_recurrent_linear_attn_delta_rule
from fla.ops.delta_rule import fused_chunk_linear_attn_delta_rule

def time_fwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean

def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_combined(func, *args, **kwargs)
    return time_fb[1].mean

repeats = 256
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = (["retnet_fused_chunk", "delta_chunk", "delta_recurrent"])

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
            v = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            fb = time_fwd_bwd(fused_chunk_retention, q, k, v,  verbose=False)
            time_f_b[config, "retnet_fused_chunk"] = fb

            q2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            v2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            f_b = time_fwd_bwd(
                fused_chunk_linear_attn_delta_rule, q, k, v, verbose=False
            )
            time_f_b[config, "delta_chunk"] = f_b


            q2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            v2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            f_b = time_fwd_bwd(
                fused_recurrent_linear_attn_delta_rule, q, k, v, verbose=False
            )
            time_f_b[config, "delta_recurrent"] = f_b

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(f"{method:>50} fwd + bwd:\t {time_f_b[config, method]*1000:>6.4f} ms ")

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

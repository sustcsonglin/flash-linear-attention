# -*- coding: utf-8 -*-

import torch
import triton
from flash_attn import flash_attn_func

from fla.ops.triton.based import parallel_based
from fla.ops.triton.retention import fused_chunk_retention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['flash', 'retention', 'based'],
        # label name for the lines
        line_names=['flash', 'retention', 'based'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'), ('yellow', 'dotted'), ('black', 'dashed')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    device = 'cuda'
    dtype = torch.bfloat16
    requires_grad = True
    batch_size, n_heads, d_head = 8, 32, 128

    if provider == 'flash':
        q = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    else:
        q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'retention':
        results = triton.testing.do_bench(lambda: fused_chunk_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'based':
        results = triton.testing.do_bench(lambda: parallel_based(q, k, v).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True, save_path='.')

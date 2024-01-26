# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.retention import (chunk_retention, fused_chunk_retention,
                               fused_recurrent_retention, parallel_retention)
from fla.ops.retention.naive import naive_retention

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['fused_chunk', 'chunk', 'parallel',
                   'fused_chunk_bwd', 'chunk_bwd', 'parallel_bwd'] + (['flash', 'flash_bwd'] if HAS_FLASH else []),
        # label name for the lines
        line_names=['fused_chunk_fwd', 'chunk_fwd', 'parallel_fwd', 'fused_chunk_fwdbwd',
                    'chunk_fwdbwd', 'parallel_fwdbwd'] + (['flash_fwd', 'flash_fwdbwd'] if HAS_FLASH else []),
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'), ('blue', 'dotted'),
                ('red', 'dotted')] + ([('cyan', '-'), ('cyan', 'dotted')] if HAS_FLASH else []),
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

    if provider == 'flash' or provider == 'flash_bwd':
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
    if provider == 'torch':
        if seq_len > 2048:
            return results
        results = triton.testing.do_bench(lambda: naive_retention(q, k, v), quantiles=quantiles)
    elif provider == 'recurrent':
        results = triton.testing.do_bench(lambda: fused_recurrent_retention(q, k, v), quantiles=quantiles)
    elif provider == 'chunk':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v), quantiles=quantiles)
    elif provider == 'fused_chunk':
        results = triton.testing.do_bench(lambda: fused_chunk_retention(q, k, v), quantiles=quantiles)
    elif provider == 'parallel':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v), quantiles=quantiles)
    elif provider == 'torch_bwd':
        if seq_len > 2048:
            return results
        results = triton.testing.do_bench(lambda: naive_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'recurrent_bwd':
        results = triton.testing.do_bench(lambda: fused_recurrent_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'chunk_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'fused_chunk_bwd':
        results = triton.testing.do_bench(lambda: fused_chunk_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'parallel_bwd':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v).backward(do), quantiles=quantiles)

    elif provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True, save_path='.')

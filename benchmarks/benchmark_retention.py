# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.triton.retention import (fused_chunk_retention, naive_retention,
                                      parallel_retention, fused_recurrent_retention, chunk_retention)


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
                   'fused_chunk_bwd', 'chunk_bwd', 'parallel_bwd'],
        # label name for the lines
        line_names=['fused_chunk', 'chunk', 'parallel', 'fused_chunk_bwd', 'chunk_bwd', 'parallel_bwd'],
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
    batch_size, n_heads, d_head_qk, d_head_v = 1, 12, 256, 432

    q = torch.randn(batch_size, n_heads, seq_len, d_head_qk, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, d_head_qk, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, d_head_v, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(v, dtype=dtype)

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
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)

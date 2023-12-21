# -*- coding: utf-8 -*-

import torch
import triton

from retention import chunk_retention, naive_retention, parallel_retention

if __name__ == '__main__':
    B, H, T, D = 2, 8, 1024, 64
    dtype = torch.float
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(q)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = chunk_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, 1e-2), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-2), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-2), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-2), breakpoint()
    print('Done!')

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['seq_len'],
            # different possible values for `x_name`
            x_vals=[128 * 2 ** i for i in range(0, 8)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            # possible values for `line_arg``
            line_vals=['torch', 'chunk', 'parallel', 'torch_bwd', 'chunk_bwd', 'parallel_bwd'],
            # label name for the lines
            line_names=['torch', 'chunk', 'parallel', 'torch_bwd', 'chunk_bwd', 'parallel_bwd'],
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
        batch_size, n_heads, d_head = 32, 16, 64

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
        elif provider == 'chunk':
            results = triton.testing.do_bench(lambda: chunk_retention(q, k, v), quantiles=quantiles)
        elif provider == 'parallel':
            results = triton.testing.do_bench(lambda: parallel_retention(q, k, v), quantiles=quantiles)
        elif provider == 'torch_bwd':
            if seq_len > 2048:
                return results
            results = triton.testing.do_bench(lambda: naive_retention(q, k, v).backward(do), quantiles=quantiles)
        elif provider == 'chunk_bwd':
            results = triton.testing.do_bench(lambda: chunk_retention(q, k, v).backward(do), quantiles=quantiles)
        elif provider == 'parallel_bwd':
            results = triton.testing.do_bench(lambda: parallel_retention(q, k, v).backward(do), quantiles=quantiles)
        return results
    benchmark.run(print_data=True)

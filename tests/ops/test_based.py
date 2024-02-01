# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.based import (fused_chunk_based, naive_parallel_based,
                           parallel_based)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [8, 15])
@pytest.mark.parametrize("T", [128, 31, 312])
def test_based(dtype, D, T, B=4, H=4):
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = (torch.randn((B, H, T, 16), dtype=dtype, device='cuda') / 10).requires_grad_()
    k = (torch.randn((B, H, T, 16), dtype=dtype, device='cuda') / 10).requires_grad_()
    v = (torch.randn((B, H, T, D), dtype=dtype, device='cuda')).requires_grad_()
    do = torch.randn_like(v) / 10
    ref = naive_parallel_based(q, k, v, True, True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = parallel_based(q, k, v, True, True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)

    tri = fused_chunk_based(q, k, v, True, True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)

    print('Done!')

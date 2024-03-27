# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.based.naive import naive_parallel_based


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [300, 512])
@pytest.mark.parametrize("hidden_size", [8, 15])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_based(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = (torch.randn((batch_size, n_heads, seq_len, 16), dtype=dtype, device='cuda') / 10).requires_grad_()
    k = (torch.randn((batch_size, n_heads, seq_len, 16), dtype=dtype, device='cuda') / 10).requires_grad_()
    v = (torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda')).requires_grad_()
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

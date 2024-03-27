# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.abc import chunk_abc
from fla.ops.abc.naive import naive_recurrent_abc


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [300, 512])
@pytest.mark.parametrize("hidden_size", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("n_slots", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    expand_ratio: int,
    n_slots: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float32 else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((batch_size, n_heads, seq_len, hidden_size * expand_ratio), dtype=dtype, device='cuda').requires_grad_()
    s = torch.randn((batch_size, n_heads, seq_len, n_slots), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref, _ = naive_recurrent_abc(q, k, v, s)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None

    # triton implementation
    tri, _ = chunk_abc(q, k, v, s)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    f"o diff: {torch.abs(ref - tri).max()}"
    f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    f"ds diff: {torch.abs(ref_ds - tri_ds).max()}"
    assert ref.allclose(tri, 0, atol)
    assert ref_dq.allclose(tri_dq, 0, atol)
    assert ref_dk.allclose(tri_dk, 0, atol)
    assert ref_dv.allclose(tri_dv, 0, atol)
    assert ref_ds.allclose(tri_ds, 0, atol)

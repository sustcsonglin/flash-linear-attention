# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [300, 512])
@pytest.mark.parametrize("hidden_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref, _ = naive_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri, _ = fused_recurrent_gla(q, k, v, g)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [300, 512])
@pytest.mark.parametrize("hidden_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_chunk(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref, _ = fused_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri, _ = chunk_gla(q, k, v, g)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [300, 512])
@pytest.mark.parametrize("hidden_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_fused_chunk(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((batch_size, n_heads, seq_len, hidden_size), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref, _ = fused_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri, _ = fused_chunk_gla(q, k, v, g)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"

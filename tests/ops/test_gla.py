# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("T", [300, 512])
def test_chunk(dtype, D, T, B=4, H=4):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref = fused_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri = chunk_gla(q, k, v, g)
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("T", [300, 512])
def test_fused_chunk(dtype, D, T, B=4, H=4):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).clamp_min(-3).requires_grad_(True)
    do = torch.randn_like(v)
    ref = fused_recurrent_gla(q, k, v, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    # triton implementation
    tri = fused_chunk_gla(q, k, v, g)
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

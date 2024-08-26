# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from fla.ops.rwkv6.recurrent_naive import (naive_recurrent_rwkv6,
                                           naive_recurrent_rwkv6_bwd)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [1000])
@pytest.mark.parametrize("D", [100])
@pytest.mark.parametrize("dtype", [torch.float])
def test_recurrent_naive(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device='cuda').to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device='cuda')).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device='cuda').to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device='cuda')
    h = torch.randn(B, H, D, 2*D, device='cuda', dtype=dtype, requires_grad=True)

    o, _ = naive_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    dq2, dk2, dv2, dw2, du2, dh2 = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do, initial_state=h)

    assert dq.allclose(dq2, atol=1e-3)
    assert dk.allclose(dk2, atol=1e-3)
    assert dv.allclose(dv2, atol=1e-3)
    assert dw.allclose(dw2, atol=1e-3)
    assert du.allclose(du2, atol=1e-3)
    assert dh.allclose(dh2, atol=1e-3)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1000])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device='cuda').to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device='cuda')).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device='cuda').to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device='cuda')
    h = torch.randn(B, H, D, 2*D, device='cuda', dtype=dtype, requires_grad=True)

    ref_o, _ = naive_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh, h.grad = h.grad.clone(), None

    assert ref_o.allclose(tri_o, atol=1e-3)
    assert ref_dq.allclose(tri_dq, atol=1e-3)
    assert ref_dk.allclose(tri_dk, atol=1e-3)
    assert ref_dv.allclose(tri_dv, atol=1e-3)
    assert ref_dw.allclose(tri_dw, atol=1e-3)
    assert ref_du.allclose(tri_du, atol=1e-3)
    assert ref_dh.allclose(tri_dh, atol=1e-3)
    assert ref_du.allclose(tri_du, atol=1e-3)
    assert ref_dh.allclose(tri_dh, atol=1e-3)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1000])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-1

    q = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device='cuda').to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device='cuda').to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device='cuda')).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device='cuda').to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device='cuda')
    h = torch.randn(B, H, D, 2*D, device='cuda', dtype=dtype, requires_grad=True)

    ref_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = chunk_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh, h.grad = h.grad.clone(), None

    assert ref_o.allclose(tri_o, atol=atol)
    assert ref_dq.allclose(tri_dq, atol=atol)
    assert ref_dk.allclose(tri_dk, atol=atol)
    assert ref_dv.allclose(tri_dv, atol=atol)
    assert ref_dw.allclose(tri_dw, atol=atol)
    assert ref_du.allclose(tri_du, atol=atol)
    assert ref_dh.allclose(tri_dh, atol=atol)
    assert ref_du.allclose(tri_du, atol=atol)
    assert ref_dh.allclose(tri_dh, atol=atol)

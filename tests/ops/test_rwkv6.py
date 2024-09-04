# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6, native_recurrent_rwkv6
from fla.ops.rwkv6.recurrent_naive import (naive_recurrent_rwkv6,
                                           naive_recurrent_rwkv6_bwd)
from fla.utils import device
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [1024])
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

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(B, H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

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
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("use_h", [False, True])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    use_h: bool
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-1

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

    ref_o, _ = native_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert ref_o.allclose(tri_o, atol=atol)
    assert ref_dq.allclose(tri_dq, atol=atol)
    assert ref_dk.allclose(tri_dk, atol=atol)
    assert ref_dv.allclose(tri_dv, atol=atol)
    assert ref_dw.allclose(tri_dw, atol=atol)
    assert ref_du.allclose(tri_du, atol=atol)
    if use_h:
        assert ref_dh.allclose(tri_dh, atol=atol)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("use_h", [False, True])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    use_h: bool
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-1

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

    ref_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = chunk_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert ref_o.allclose(tri_o, atol=atol)
    assert ref_dq.allclose(tri_dq, atol=atol)
    assert ref_dk.allclose(tri_dk, atol=atol)
    assert ref_dv.allclose(tri_dv, atol=atol)
    assert ref_dw.allclose(tri_dw, atol=atol)
    assert ref_du.allclose(tri_du, atol=atol)
    if use_h:
        assert ref_dh.allclose(tri_dh, atol=atol)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("HEAD_SIZE", [64])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_chunk_error_ratio(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype
):
    atol = 1e-3 if dtype == torch.float else 1e-2
    H = C // HEAD_SIZE
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / base

    def val(x):
        return x.detach().float().cpu().numpy()

    def LOSS(y):
        return ((y * y) - torch.tanh(y)).sum()

    def RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, h):
        r = r.view(B,T,H,-1).transpose(1,2)
        k = k.view(B,T,H,-1).transpose(1,2)
        v = v.view(B,T,H,-1).transpose(1,2)
        w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
        o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=False)
        return o.transpose(1,2).reshape(B,T,C)

    def RUN_FLA_FUSED(B, T, C, H, r, k, v, w, u, h):
        r = r.view(B,T,H,-1).transpose(1,2)
        k = k.view(B,T,H,-1).transpose(1,2)
        v = v.view(B,T,H,-1).transpose(1,2)
        w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
        o,_ = fused_recurrent_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=False)
        return o.transpose(1,2).reshape(B,T,C)

    set_seed(42)
    with torch.no_grad():
        r = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        w = torch.empty(B, T, C, device=device).uniform_(-8, 1).to(dtype=dtype)
        u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype)
        initial_state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=device).to(dtype=dtype)

    def clear_grad():
        r.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        w.requires_grad_()
        u.requires_grad_()
        initial_state.requires_grad_()
        if r.grad is not None: r.grad.data.zero_()
        if k.grad is not None: k.grad.data.zero_()
        if v.grad is not None: v.grad.data.zero_()
        if w.grad is not None: w.grad.data.zero_()
        if u.grad is not None: u.grad.data.zero_()
        if initial_state.grad is not None: initial_state.grad.data.zero_()

    clear_grad()
    y32 = RUN_FLA_FUSED(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    LOSS(y32).backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
    clear_grad()

    yF16 = RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, initial_state)
    LOSS(yF16).backward()
    gr_chunk = r.grad.data.clone()
    gk_chunk = k.grad.data.clone()
    gv_chunk = v.grad.data.clone()
    gw_chunk = w.grad.data.clone()
    gu_chunk = u.grad.data.clone()
    clear_grad()

    assert get_err_ratio(yF16, y32) < atol
    assert get_err_ratio(gr_chunk, gr) < atol
    assert get_err_ratio(gk_chunk, gk) < atol
    assert get_err_ratio(gv_chunk, gv) < atol
    assert get_err_ratio(gw_chunk, gw) < atol
    assert get_err_ratio(gu_chunk, gu) < atol

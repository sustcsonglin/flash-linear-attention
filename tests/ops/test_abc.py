# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.abc import chunk_abc, fused_recurrent_gated_abc
from fla.ops.abc.naive import naive_recurrent_abc


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("V", [64, 128, 200])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    g = F.logsigmoid(torch.randn((B, H, T, M), dtype=dtype, device='cuda')).requires_grad_()
    do = torch.randn_like(v)
    ref, _ = naive_recurrent_abc(q, k, v, s, g)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None

    tri, _ = fused_recurrent_gated_abc(q, k, v, s, g)
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


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("V", [64, 128, 200])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref, _ = naive_recurrent_abc(q, k, v, s)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None

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


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("V", [64, 128, 200])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_inference(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float else 1e-1

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda')
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda')
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda')
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda')
    g = F.logsigmoid(torch.randn((B, H, T, M), dtype=dtype, device='cuda'))

    ref, _ = fused_recurrent_gated_abc(q, k, v, s, g)
    tri = torch.empty_like(ref)
    ht = (q.new_zeros(B, H, K, M), q.new_zeros(B, H, M, V))
    for i in range(T):
        o, ht = fused_recurrent_gated_abc(
            q[:, :, i:i+1],
            k[:, :, i:i+1],
            v[:, :, i:i+1],
            s[:, :, i:i+1],
            g[:, :, i:i+1],
            initial_state=ht,
            output_final_state=True
        )
        tri[:, :, i] = o.squeeze(2)
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"

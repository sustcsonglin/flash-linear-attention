# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.gsa.naive import naive_recurrent_gsa


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, atol):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert ref.allclose(tri, 0, atol), msg


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 100])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3

    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    g = F.logsigmoid(torch.randn((B, H, T, M), dtype=dtype, device='cuda')).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device='cuda').requires_grad_()
    hv0 = torch.randn(B, H, M, D, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(q, k, v, s, g, initial_state=(hk0, hv0), output_final_state=True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dhk0, hk0.grad = hk0.grad.clone(), None
    ref_dhv0, hv0.grad = hv0.grad.clone(), None

    tri, (tri_hkt, tri_hvt) = fused_recurrent_gsa(q, k, v, s, g, initial_state=(hk0, hv0), output_final_state=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    tri_dg, s.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close("   o", ref, tri, atol)
    assert_close(" hkt", ref_hkt, tri_hkt, atol)
    assert_close(" hvt", ref_hvt, tri_hvt, atol)
    assert_close("  dq", ref_dq, tri_dq, atol)
    assert_close("  dk", ref_dk, tri_dk, atol)
    assert_close("  dv", ref_dv, tri_dv, atol)
    assert_close("  ds", ref_ds, tri_ds, atol)
    assert_close("  dg", ref_dg, tri_dg, atol)
    assert_close("dhk0", ref_dhk0, tri_dhk0, atol)
    assert_close("dhv0", ref_dhv0, tri_dhv0, atol)


@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [64, 100, 512])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-1
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    g = F.logsigmoid(torch.randn((B, H, T, M), dtype=dtype, device='cuda')).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device='cuda').requires_grad_()
    hv0 = torch.randn(B, H, M, D, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    ref, _ = fused_recurrent_gsa(q, k, v, s, g, initial_state=(hk0, hv0))
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = chunk_gsa(q, k, v, s, g, initial_state=(hk0, hv0))
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    tri_dg, s.grad = g.grad.clone(), None

    assert_close(" o", ref, tri, atol)
    assert_close("dq", ref_dq, tri_dq, atol)
    assert_close("dk", ref_dk, tri_dk, atol)
    assert_close("dv", ref_dv, tri_dv, atol)
    assert_close("ds", ref_ds, tri_ds, atol)
    assert_close("dg", ref_dg, tri_dg, atol)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("HQ", [8, 16])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 100])
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_inference(
    B: int,
    HQ: int,
    H: int,
    T: int,
    D: int,
    M: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3

    q = torch.randn((B, HQ, T, D), dtype=dtype, device='cuda')
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    s = torch.randn((B, H, T, M), dtype=dtype, device='cuda')
    g = F.logsigmoid(torch.randn((B, H, T, M), dtype=dtype, device='cuda'))
    h0 = (torch.zeros(B, H, D, M, dtype=dtype, device='cuda'),
          torch.zeros(B, H, M, D, dtype=dtype, device='cuda'))

    ref, _ = naive_recurrent_gsa(q, k, v, s, g, initial_state=h0)
    tri = torch.empty_like(ref)
    for i in range(T):
        o, ht = fused_recurrent_gsa(
            q[:, :, i:i+1],
            k[:, :, i:i+1],
            v[:, :, i:i+1],
            s[:, :, i:i+1],
            g[:, :, i:i+1],
            initial_state=h0,
            output_final_state=True
        )
        tri[:, :, i] = o.squeeze(2)
        assert_close(f"o{i}", ref[:, :, i], tri[:, :, i], atol)
        h0 = ht

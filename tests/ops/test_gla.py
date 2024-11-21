# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla, fused_recurrent_gla


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [130, 146, 162, 178, 300, 2048])
@pytest.mark.parametrize("D", [300, 100])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("head_first", [True, False])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    head_first: bool
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # [B, H, T, D]
    if head_first:
        q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        g = F.logsigmoid(torch.randn((B, H, T, D), dtype=dtype, device='cuda')).requires_grad_()
    else:
        q = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        g = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device='cuda')).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    dht = torch.zeros((B, H, D, D), dtype=dtype, device='cuda')

    tri, tri_ht = chunk_gla(q, k, v, g, initial_state=h0, output_final_state=True, head_first=head_first)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    ref, ref_ht = fused_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=True, head_first=head_first)
    ref, _ = fused_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=False, head_first=head_first)
    (ref * do).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    assert_close("  o", ref, tri, 0.004)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.005)
    assert_close(" dk", ref_dk, tri_dk, 0.005)
    assert_close(" dv", ref_dv, tri_dv, 0.005)
    assert_close(" dg", ref_dg, tri_dg, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.005)

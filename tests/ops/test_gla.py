# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla
from fla.ops.gla.naive import naive_recurrent_gla


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("T", [130, 146, 162, 178, 300, 2048])
@pytest.mark.parametrize("D", [300, 100])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    # [B, H, T, D]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    g = F.logsigmoid(torch.randn((B, H, T, D), dtype=dtype, device='cuda')).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((B, H, D, D), dtype=dtype, device='cuda')

    tri, tri_ht = chunk_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    ref, ref_ht = naive_recurrent_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    assert get_err_ratio(tri, ref) < 0.004, f" o diff: {get_abs_err(ref, tri)}, ratio: {get_err_ratio(ref, tri)}"
    assert get_err_ratio(tri_ht, ref_ht) < 0.005, \
        f"ht diff: {get_abs_err(ref_ht, tri_ht)}, ratio: {get_err_ratio(ref_ht, tri_ht)}"
    assert get_err_ratio(tri_dq, ref_dq) < 0.005, \
        f"dq diff: {get_abs_err(ref_dq, tri_dq)}, ratio: {get_err_ratio(ref_dq, tri_dq)}"
    assert get_err_ratio(tri_dk, ref_dk) < 0.005, \
        f"dk diff: {get_abs_err(ref_dk, tri_dk)}, ratio: {get_err_ratio(ref_dk, tri_dk)}"
    assert get_err_ratio(tri_dv, ref_dv) < 0.005, \
        f"dv diff: {get_abs_err(ref_dv, tri_dv)}, ratio: {get_err_ratio(ref_dv, tri_dv)}"
    assert get_err_ratio(tri_dg, ref_dg) < 0.005, \
        f"dg diff: {get_abs_err(ref_dg, tri_dg)}, ratio: {get_err_ratio(ref_dg, tri_dg)}"
    assert get_err_ratio(tri_dh0, ref_dh0) < 0.005, \
        f"dh0 diff: {get_abs_err(ref_dh0, tri_dh0)}, ratio: {get_err_ratio(ref_dh0, tri_dh0)}"

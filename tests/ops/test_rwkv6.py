# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6
from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6


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


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("T", [130, 146, 162, 178])
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
    # [B, H, T, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, 2*D), dtype=dtype, device='cuda').requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = F.logsigmoid(g).requires_grad_(True)
    u = torch.randn(H, D, device='cuda').to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device='cuda').to(dtype).requires_grad_(True)
    h0 = torch.randn((B, H, D, 2*D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((B, H, D, 2*D), dtype=dtype, device='cuda')
    ref, ref_ht = naive_recurrent_rwkv6(q.clone(),
                                        k.clone(),
                                        v.clone(),
                                        g.clone(),
                                        u.clone(),
                                        initial_state=h0.clone(),
                                        output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None

    # triton implementation
    tri, tri_ht = chunk_rwkv6(q, k, v, g, u, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None

    assert_close('  o', tri, ref, 0.004)
    assert_close(' ht', tri_ht, ref_ht, 0.005)
    assert_close(' dq', tri_dq, ref_dq, 0.005)
    assert_close(' dk', tri_dk, ref_dk, 0.005)
    assert_close(' dv', tri_dv, ref_dv, 0.005)
    assert_close(' dg', tri_dg, ref_dg, 0.005)
    assert_close(' du', tri_du, ref_du, 0.005)
    assert_close('dh0', tri_dh0, ref_dh0, 0.005)

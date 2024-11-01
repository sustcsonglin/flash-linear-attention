# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.retention import (chunk_retention, fused_recurrent_retention,
                               parallel_retention)
from fla.ops.retention.naive import naive_retention


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_chunk(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float32 else 1e-1
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ref, ref_ht = fused_recurrent_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = chunk_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"
    assert ref_ht.allclose(tri_ht, 0, atol), f"o diff: {torch.abs(ref_ht - tri_ht).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_parallel(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float32 else 1e-1
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"

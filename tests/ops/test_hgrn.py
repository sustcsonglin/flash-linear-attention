# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn
from fla.ops.hgrn.naive import naive_recurrent_hgrn


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
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
    atol = 1e-4

    x = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, :, 0])
    ref, _ = naive_recurrent_hgrn(x, g, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dx.allclose(tri_dx, 0, atol), f"dx diff: {torch.abs(ref_dx - tri_dx).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-2

    x = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, :, 0])
    ref, _ = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = chunk_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dx.allclose(tri_dx, 0, atol), f"dx diff: {torch.abs(ref_dx - tri_dx).max()}"
    assert ref_dg.allclose(tri_dg, 0, atol), f"dg diff: {torch.abs(ref_dg - tri_dg).max()}"

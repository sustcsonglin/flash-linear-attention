# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.triton.gla import chunk_gla, fused_chunk_gla
from torch.nn.functional import logsigmoid

@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("T", [512])
def test_gla(dtype, D, T, B=4, H=4):
    # [batch_size, n_heads, seq_len, d_head]
    q = (torch.randn((B, H, T, D), dtype=dtype, device='cuda') / 10).requires_grad_()
    k = (torch.randn((B, H, T, D), dtype=dtype, device='cuda') / 10).requires_grad_()
    v = (torch.randn((B, H, T, D), dtype=dtype, device='cuda')).requires_grad_()
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = logsigmoid(g).clamp_min(-3)
    g.requires_grad_(True)
    do = torch.randn_like(v) 
    ref = chunk_gla(q, k, v, g)
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

    assert ref.allclose(tri, 0, 1e-4)
    assert ref_dq.allclose(tri_dq, 0, 1e-4)
    assert ref_dk.allclose(tri_dk, 0, 1e-4)
    assert ref_dv.allclose(tri_dv, 0, 1e-4)
    assert ref_dg.allclose(tri_dg, 0, 1e-4)

    print('Done!')


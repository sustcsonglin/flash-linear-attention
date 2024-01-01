# -*- coding: utf-8 -*-

import torch
import triton
import pytest
from fla.ops.triton.retention import fused_chunk_retention, naive_retention, parallel_retention


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [64, 100, 128,256])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("T", [1000, 1024])
def test_retention(dtype, D, expand_ratio, T, B=4, H=4):
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = (torch.randn((B, H, T, D), dtype=dtype, device='cuda') / 10).requires_grad_() 
    k = (torch.randn((B, H, T, D), dtype=dtype, device='cuda') / 10).requires_grad_()
    v = (torch.randn((B, H, T, D * expand_ratio), dtype=dtype, device='cuda')).requires_grad_()
    do = torch.randn_like(v) / 10
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = fused_chunk_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, 1e-2) 
    assert ref_dq.allclose(tri_dq, 0, 1e-2) 
    assert ref_dk.allclose(tri_dk, 0, 1e-2) 
    assert ref_dv.allclose(tri_dv, 0, 1e-2) 

    print('Done!')


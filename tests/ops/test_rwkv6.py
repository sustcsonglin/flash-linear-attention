# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6.recurrent_naive import (naive_recurrent_rwkv6,
                                           naive_recurrent_rwkv6_bwd)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seq_len", [1000])
@pytest.mark.parametrize("hidden_size", [100])
@pytest.mark.parametrize("dtype", [torch.float])
def test_recurrent_naive(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn(batch_size, n_heads, seq_len, hidden_size, device='cuda').to(dtype).requires_grad_(True)
    k = torch.randn(batch_size, n_heads, seq_len, hidden_size, device='cuda').to(dtype).requires_grad_(True)
    v = torch.randn(batch_size, n_heads, seq_len, 2*hidden_size, device='cuda').to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(batch_size, n_heads, seq_len, hidden_size, device='cuda')).to(dtype).requires_grad_(True)
    u = torch.randn(n_heads, hidden_size, device='cuda').to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device='cuda')
    h = torch.randn(batch_size, n_heads, hidden_size, 2*hidden_size, device='cuda', dtype=dtype, requires_grad=True)

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

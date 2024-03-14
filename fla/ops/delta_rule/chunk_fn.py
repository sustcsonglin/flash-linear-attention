# -*- coding: utf-8 -*-
import torch
import math
from einops import rearrange
from fla.ops.delta_rule.triton_fn import prepare_wy_repr
from fla.ops.delta_rule.chunk_fused import fused_chunk_delta_rule
from fla.ops.delta_rule.chunk import chunk_delta_rule

def chunk_linear_attn_delta_rule(q, k, v, beta=None, chunk_size=32, fused_chunk=False):
    b, h, l, d_k = q.shape
    if beta is None:
      beta = q.new_ones(b, h, l)
    # l should be multiple of chunk_size
    assert l % chunk_size == 0
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "q, k and v should have the same dimension."
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    k_cumdecay, v_new = prepare_wy_repr(k, v, beta, chunk_size)
    if fused_chunk:
        o, _ = fused_chunk_delta_rule(q, k, v_new, k_cumdecay, chunk_size)
    else:
        o, _ = chunk_delta_rule(q, k, v_new, k_cumdecay, chunk_size)
    return o 

def delta_rule_recurrence(q, k, v, beta):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o

from fla.ops.delta_rule.recurrent_fuse import fused_recurrent_linear_attn_delta_rule

if __name__ == "__main__":
    seq_len = 128
    b = 2
    h = 4
    k = torch.randn(b, h, seq_len, 64) / 10
    v = torch.randn(b, h, seq_len, 64)  
    q = torch.randn(b, h, seq_len, 64)
    beta = torch.rand(b, h, seq_len).sigmoid()
    # beta = torch.ones(b, h, seq_len)

    q, k, v, beta = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta))
    do = torch.rand_like(v)

    o2 = delta_rule_recurrence(q, k, v.clone(), beta)
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    o = chunk_linear_attn_delta_rule(q, k, v, beta)
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None
    print((o- o2).abs().max())
    assert (o- o2).abs().max() < 1e-5
    print((q_grad - q_grad2).abs().max())
    print((k_grad - k_grad2).abs().max())
    print((v_grad - v_grad2).abs().max())
    print((beta_grad - beta_grad2).abs().max())
    breakpoint()


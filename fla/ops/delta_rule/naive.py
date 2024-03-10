# -*- coding: utf-8 -*-
import re
import math
import torch


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

    
def delta_rule_backward(q, k, v, do):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * d_k ** -0.5
    v_modified = torch.empty_like(v)

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        v_modified[:, :, i] = _v
        S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)

    b, h, l, d_k = q.shape
    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)
    DS = torch.zeros_like(S)


    for i in range(l-1, -1, -1):
        do_i = do[:, :, i]
        k_i = k[:, :, i]
        q_i = q[:, :, i]
        v_i = v_modified[:, :, i]

        DS_i = q_i.unsqueeze(-1) * do_i.unsqueeze(-2)

        DS += DS_i
        dv_i = torch.einsum('bhnm, bhn -> bhm', DS, k_i)
        d_v[:, :, i] = dv_i

        dk_i = torch.einsum('bhnm, bhm -> bhn', DS, v_i)
        d_k[:, :, i] = dk_i
        DS -= dv_i.unsqueeze(-2) * k_i.unsqueeze(-1)

    S = torch.zeros_like(DS)
    for i in range(l):
        _k = k[:, :, i]
        _do = do[:, :, i]
        _v = v_modified[:, :, i]
        S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        d_q[:, :, i] = torch.einsum('bhm,bhdm->bhd', _do, S) * (q.shape[-1] ** -0.5)
        if i < l-1:
            _dv = d_v[:, :, i+1]
            d_k[:,:,i+1] -= torch.einsum('bhm,bhdm->bhd', _dv, S)
    return d_q, d_k, d_v        
        

# chunk size will be 2**scan_depth.
# by default the chunk size is 2**7=256
def delta_rule_scan(q, k, v, beta=None, scan_depth=7):
  b, h, l, d = v.shape
  q = q * (q.shape[-1] ** -0.5)
  # l should be power of 2
  assert l & (l - 1) == 0
  k_cumsum = torch.zeros_like(k)
  k_cum_decay = k.clone()

  chunk_size = 1
  step = min(scan_depth, int(math.log2(l)))

  if beta is not None:
    v = v * beta[..., None]
    k_cum_decay = k_cum_decay * beta[..., None]
  
  for i in range(step):
    k = k.view(b, h, l//chunk_size, chunk_size, -1)
    v = v.view(b, h, l//chunk_size, chunk_size, -1)
    k_cum_decay = k_cum_decay.view(b, h, l//chunk_size, chunk_size, -1)
    k_cumsum = k_cumsum.view(b, h, l//chunk_size, chunk_size, -1)
    
    k_first = k[:, :, 0::2]
    k_second = k[:, :, 1::2]
    v_first = v[:, :, 0::2]

    k_cum_decay_first = k_cum_decay[:, :, 0::2] 
    k_cum_decay_second = k_cum_decay[:, :, 1::2]
    k_cumsum_first = k_cumsum[:, :, 0::2]

    tmp = k_cum_decay_second @ k_first.transpose(-1, -2)
    next_k_cum_decay_second = -tmp @ k_cum_decay_first

    if i > 0:
      next_k_cumsum = tmp @ (-k_cumsum_first + v_first)
    else:
      next_k_cumsum = tmp @ v_first

    k_cumsum[:, :, 1::2].add_(next_k_cumsum)
    k_cum_decay[:, :, 1::2].add_(next_k_cum_decay_second)

    k = k.view(b, h, l, -1)
    v = v.view(b, h, l, -1)
    k_cum_decay = k_cum_decay.view(b, h, l, -1)
    k_cumsum = k_cumsum.view(b, h, l, -1)
    chunk_size *= 2

  # breakpoint()

    S =  k[:, :, :chunk_size].transpose(-1, -2) @ (v[:, :, :chunk_size]-k_cumsum[:, :, :chunk_size]) 
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    o = torch.empty_like(v)
    o[:, :, :chunk_size] = (q[:, :, :chunk_size] @ k[:, :, :chunk_size].transpose(-1, -2)).masked_fill_(mask, 0) @ (v[:, :, :chunk_size]-k_cumsum[:, :, :chunk_size]) 

  for i in range(1, l // chunk_size):
      ## intra chunk.
      idx = range(i*chunk_size, (i+1)*chunk_size)
      q_i, k_i, v_i = q[:, :, idx], k[:, :, idx], v[:, :, idx]
      k_cumsum_i = k_cumsum[:, :, idx]
      attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
      o_i = attn @ (v_i - k_cumsum_i) 
      ## inter chunk.
      v_prime = torch.einsum('b h m n, b h x m -> b h x n', S, k_cum_decay[:, :, idx])
      o[:, :, idx] = o_i - attn @ v_prime + q_i @ S
      ## chunk state update
      S = S + k_i.transpose(-1, -2) @ (v_i - k_cumsum_i - v_prime) 
  return S, o


def delta_rule_recurrence_no_materialize(q, k, v, beta):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    k_cumsum = torch.zeros_like(k)
    if beta is not None:
      v = v * beta[..., None]
    mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=q.device), diagonal=0)
    attn = (k @ k.transpose(-1, -2)).masked_fill_(mask, 0)
    o1 = attn @ v

    for i in range(1, l):
        k_cumsum_ = k_cumsum[:, :, :i]
        k_cumsum[:, :, i] = (-(attn[:, :, i, :i, None] * k_cumsum_).sum(2) + o1[:, :, i]) * beta[:, :, i, None]
    
    S =  k.transpose(-1, -2) @ (v - k_cumsum)
    mask = torch.triu(torch.ones(l, l, dtype=torch.bool, device=q.device), diagonal=1)
    o = (q @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ (v - k_cumsum)
    return S, o

 

if __name__ == '__main__':
    B = 2
    H = 4
    L = 128
    DK = 64
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True) 
    k = (torch.randn(B, H, L, DK) / 10).cuda().requires_grad_(True) 
    v = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True) 
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    
    o = delta_rule_recurrence(q, k, v, beta).cuda()
    do = torch.randn(B, H, L, DK).cuda()
    o.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    from fla.ops.delta_rule.recurrent_fuse import \
        fused_recurrent_linear_attn_delta_rule as triton_delta_rule
    o2 = triton_delta_rule(q, k, v, beta)
    o2.backward(do)
    assert torch.allclose(o, o2, atol=1e-4), breakpoint()
    assert torch.allclose(q.grad, q_grad, atol=1e-4), breakpoint()
    assert torch.allclose(k.grad, k_grad, atol=1e-4), breakpoint()
    assert torch.allclose(v.grad, v_grad, atol=1e-4), breakpoint()
    assert torch.allclose(beta.grad, beta_grad, atol=1e-4), breakpoint()

    print("All pass.")

    seq_len = 128
    b = 2
    h = 4
    k = torch.randn(b, h, seq_len, 64) / 10
    v = torch.randn(b, h, seq_len, 64)  
    q = torch.randn(b, h, seq_len, 64)
    beta = torch.rand(b, h, seq_len).sigmoid()
    # beta = torch.ones(b, h, seq_len)

    q, k, v, beta = map(lambda x: x.cuda(), (q, k, v, beta))

    S, o = delta_rule_scan(q, k.clone(), v.clone(), beta)
    
    from fla.ops.delta_rule.recurrent_fuse import fused_recurrent_linear_attn_delta_rule
    
    o2 = fused_recurrent_linear_attn_delta_rule(q, k.clone(), v.clone(), beta)

    print((o- o2).abs().max())
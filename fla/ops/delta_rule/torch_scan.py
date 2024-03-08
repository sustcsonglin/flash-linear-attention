# -*- coding: utf-8 -*-
import torch
import math

# chunk size will be 2**scan_depth.
# by default the chunk size is 2**7=128
def delta_rule_scan(q, k, v, beta, scan_depth=7):
  b, h, l, d = v.shape
  q = q * (q.shape[-1] ** -0.5)
  # l should be power of 2
  assert l & (l - 1) == 0

  k_cumsum = torch.zeros_like(k)
  k_cum_decay = k.clone()

  chunk_size = 1
  step = min(scan_depth, int(math.log2(l)))

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

    next_k_cum_decay_second = k_cum_decay_second + k_cum_decay_second @ (-k_first).transpose(-1, -2) @ k_cum_decay_first

    if i > 0:
      next_k_cumsum = (k_cum_decay_second @ -k_first.transpose(-1, -2)) @ k_cumsum_first + (k_cum_decay_second @ k_first.transpose(-1, -2)) @ v_first
    else:
      next_k_cumsum = (k_cum_decay_second @ k_first.transpose(-1, -2)) @ v_first

    k_cumsum[:, :, 1::2] += next_k_cumsum
    k_cum_decay[:, :, 1::2] = next_k_cum_decay_second

    k = k.view(b, h, l, -1)
    v = v.view(b, h, l, -1)
    k_cum_decay = k_cum_decay.view(b, h, l, -1)
    k_cumsum = k_cumsum.view(b, h, l, -1)

    chunk_size *= 2

  S =  k[:, :, :chunk_size].transpose(-1, -2) @ (v[:, :, :chunk_size]-k_cumsum[:, :, :chunk_size]) 
  mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
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
    return S, o


if __name__ == "__main__":
    seq_len = 1024
    b = 2
    h = 4
    k = torch.randn(b, h, seq_len, 32) / 10
    v = torch.randn(b, h, seq_len, 32) / 10
    q = torch.randn(b, h, seq_len, 32)
    beta = torch.rand(b, h, seq_len).sigmoid()

    o2, y2  = delta_rule_scan(q, k.clone(), v.clone(), beta)
    o3, y3 = delta_rule_recurrence(q, k.clone(), v.clone(), beta)
    # o4, y4 = cumulative_product_fast_v3(u.clone(), v.clone(), u2.clone(), v2.clone(), q)
    # print((o2-o3).abs().max())
    print((o3-o2).abs().max())
    print((y3-y2).abs().max())

# -*- coding: utf-8 -*-

import torch


def naive_abc_attention(q, k, v, sk, sv):
    *_, d_head = q.shape
    dtype, scale = q.dtype, d_head ** -0.5
    # [batch_size, n_heads, seq_len, n_slots]
    sk = (sk - sk.max(2, True)[0]).to(torch.float).exp()
    sv = (sv - sv.max(2, True)[0]).to(torch.float).exp()
    zk, zv = sk.cumsum(2), sv.cumsum(2)
    sk, zk = (sk / zk[:, :, -1:]).to(dtype), (zk / zk[:, :, -1:]).to(dtype)
    sv, zv = (sv / zv[:, :, -1:]).to(dtype), (zv / zv[:, :, -1:]).to(dtype)
    # [batch_size, n_heads, seq_len, n_slots, d_head]
    K = (sk.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / zk.unsqueeze(-1)
    V = (sv.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / zv.unsqueeze(-1)
    # [batch_size, n_heads, seq_len, n_slots]
    p = torch.einsum('...d,...md->...m', q * scale, K).softmax(-1, dtype=torch.float).to(dtype)
    # [batch_size, n_heads, seq_len, d_head]
    o = torch.einsum('...m,...md->...d', p, V)
    return o

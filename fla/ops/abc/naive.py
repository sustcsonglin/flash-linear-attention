# -*- coding: utf-8 -*-

import torch


def naive_abc(q, k, v, s):
    # no numerical stability guarantees, only for tests
    *_, d_head = q.shape
    dtype, scale = q.dtype, d_head ** -0.5
    # [batch_size, n_heads, seq_len, n_slots]
    s = (s - s.max(2, True)[0]).to(torch.float).exp()
    z = s.cumsum(2)
    s, z = s.to(dtype), z.to(dtype)
    # [batch_size, n_heads, seq_len, n_slots, d_head]
    K = (s.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    V = (s.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    # [batch_size, n_heads, seq_len, n_slots]
    p = torch.einsum('...d,...md->...m', q * scale, K).softmax(-1, dtype=torch.float).to(dtype)
    # [batch_size, n_heads, seq_len, d_head]
    o = torch.einsum('...m,...md->...d', p, V)
    return o

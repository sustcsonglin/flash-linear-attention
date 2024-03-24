# -*- coding: utf-8 -*-

from typing import Optional

import torch


def naive_recurrent_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> torch.Tensor:
    dtype = q.dtype
    q, k, v, s = map(lambda x: x.float(), (q, k, v, s))
    # [batch_size, n_heads, seq_len, n_slots]
    z = s.logcumsumexp(2)
    g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
    B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    hk = torch.zeros(B, H, K, M, dtype=torch.float, device=q.device)
    ok = torch.zeros_like(s)
    scale = K ** -0.5

    if initial_state is not None:
        hk += initial_state

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = s[:, :, i]
        z_i = z[:, :, i]
        g_i = g[:, :, i].exp()
        hk = hk * g_i[..., None, :] + k_i[..., None] * (v_i - z_i).exp()[..., None, :]
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)
    p = ok.softmax(-1)

    hv = torch.zeros(B, H, M, V, dtype=torch.float, device=q.device)
    ov = torch.zeros_like(v)
    for i in range(T):
        q_i = p[:, :, i]
        k_i = s[:, :, i]
        v_i = v[:, :, i]
        z_i = z[:, :, i]
        g_i = g[:, :, i].exp()
        hv = hv * g_i[..., :, None] + (k_i - z_i).exp()[..., None] * v_i[..., None, :]
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

    return ov.to(dtype), hv

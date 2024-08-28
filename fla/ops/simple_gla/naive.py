# -*- coding: utf-8 -*-

import torch
from einops import rearrange


def torch_simple_gla(q, k, v, g, chunk_size=64, scale=None):
    if scale is None:
        scale = (q.shape[-1] ** -0.5)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    g = rearrange(g, 'b h (n c) -> b h n c', c=chunk_size)
    g = g.cumsum(-1)
    kv = k.transpose(-1, -2) @ (v * (-g + g[:, :, :, -1, None]).exp()[..., None])
    S = torch.zeros_like(kv)

    for i in range(1, g.shape[-2]):
        S[:, :, i] = S[:, :, i-1].clone() * g[:, :, i-1, -1, None, None].exp() + kv[:, :, i-1]

    inter = (q * g[..., None].exp()) @ S
    attn = q @ k.transpose(-1, -2)
    attn = attn * (g[..., None] - g[..., None, :]).exp()
    attn = attn.masked_fill(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)
    intra = attn @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')


def torch_simple_gla_recurrent(q, k, v, g, initial_state=None, scale=None):
    B, H, T, DK = q.shape
    if scale is None:
        scale = DK ** -0.5
    q = q * scale
    _, _, _, DV = v.shape
    if initial_state is None:
        S = torch.zeros(B, H, DK, DV).to(q)
    else:
        S = initial_state
    o = torch.zeros(B, H, T, DV).to(q)
    for i in range(T):
        gate = g[:, :, i].exp()
        key = k[:, :, i]
        value = v[:, :, i]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        S = S.clone() * gate.unsqueeze(-1).unsqueeze(-1) + kv
        q_i = q[:, :, i, :]
        o_i = (q_i.unsqueeze(-1) * S).sum(-2)
        o[:, :, i] = o_i
    return o, S

if __name__ == '__main__':
    torch.set_default_dtype(torch.bfloat16)
    B = 4
    H = 4
    L = 100
    DK = 32
    DV = 32
    q = torch.randn(B, H, L, DK)
    k = torch.randn(B, H, L, DK)
    v = torch.randn(B, H, L, DV)
    g = torch.nn.functional.logsigmoid(torch.randn(B, H, L))
    q, k, v, g = map(lambda x: x.cuda().requires_grad_(True), [q, k, v, g])
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla

    o, _ = fused_recurrent_simple_gla(q, k, v, g)
    do = torch.randn_like(o)
    o.backward(do)
    q_grad, k_grad, v_grad, g_grad = q.grad, k.grad, v.grad, g.grad
    q.grad, k.grad, v.grad, g.grad = None, None, None, None
    o2, _ = chunk_simple_gla(q, k, v, g)
    o2.backward(do)
    q_grad2, k_grad2, v_grad2, g_grad2 = q.grad, k.grad, v.grad, g.grad

    print((o-o2).abs().max())
    print((q_grad-q_grad2).abs().max())
    print((k_grad-k_grad2).abs().max())
    print((v_grad-v_grad2).abs().max())
    print((g_grad-g_grad2).abs().max())



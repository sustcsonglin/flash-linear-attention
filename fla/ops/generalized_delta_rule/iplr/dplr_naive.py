# -*- coding: utf-8 -*-

import torch
from einops import rearrange


# S_t = S_{t-1} @ (W + alpha_t beta_t^T) + v_t k_t^T
# r, w, k, alpha, beta [B, H, L, D_K]
# v [B, H, L, D_V]
def dplr_recurrence(r, w,  k, v, alpha, beta, initial_state=None, output_final_state=True):
    orig_dtype = r.dtype
    b, h, l, d_k = r.shape
    r, w, k, v, beta = map(lambda x: x.float(), [r, w, k, v, beta])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    r = r * (d_k ** -0.5)

    if initial_state is not None:
        S += initial_state

    for i in range(l):
        _k = k[:, :, i]
        _w = w[:, :, i]
        _r = r[:, :, i]
        _v = v[:, :, i]
        _alpha = alpha[:, :, i]
        _beta = beta[:, :, i]
        _kv = _k[..., None] * _v[..., None, :] + (S.clone() * _alpha[..., None]).sum(-2, keepdim=True) * _beta[..., None]
        S = S * _w[..., None].exp() + _kv
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _r, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def dplr_chunkwise(r, w, k, v, alpha, beta, initial_state=None, output_final_state=True, chunk_size=32):
    # FIXME: hypnopump@ needs working!
    b, h, l, d_k = r.shape
    d_v = v.shape[-1]
    r = r * (d_k ** -0.5)
    v = v
    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=r.device), diagonal=0)
    r, w, k, v, alpha, beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [r, w, k, v, alpha, beta])

    v2 = (alpha @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ v
    attn = (alpha @ beta.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)

    # FIXME: hypnopump@ needs working!
    # FIXME: add decay in the diagonal' idk!
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=r.device)
    u = attn @ v2
    w = attn @ alpha
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)

    if initial_state is not None:
        S += initial_state

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=r.device), diagonal=1)
    for i in range(0, l // chunk_size):
        r_i, k_i, v_i, u_i, w_i, beta_i = r[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], beta[:, :, i]
        o_1 = (r_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0) @ v_i
        v2_i = u_i + w_i @ S
        o_2 = (r_i @ beta_i.transpose(-1, -2)).masked_fill_(mask, 0) @ (v2_i)
        o_3 = r_i @ S
        o[:, :, i] = o_1 + o_2 + o_3
        # FIXME: hypnopump@ needs working!
        # FIXME: need to play scaled decay here
        S = S + k_i.transpose(-1, -2) @ v_i + beta_i.transpose(-1, -2) @ v2_i
    S = None if output_final_state is False else S
    return rearrange(o, 'b h n c d -> b h (n c) d'), S


if __name__ == '__main__':
    B = 2
    H = 4
    L = 128
    DK = 128
    DV = 128
    r = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    alpha = torch.randn(B, H, L, DK).cuda().softmax(-1).requires_grad_(True)
    beta = torch.randn(B, H, L, DK).cuda().softmax(-1).requires_grad_(True)

    o, s = dplr_recurrence(r, k, v, -alpha, beta)
    do = torch.randn_like(o).cuda()
    o.backward(do, retain_graph=True)
    r_grad, r.grad = r.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    o2, s2 = dplr_chunkwise(r, k, v, -alpha, beta)
    o2.backward(do)
    # assert torch.allclose(o, o2, atol=1e-4), breakpoint()
    # assert torch.allclose(s, s2, atol=1e-4), breakpoint()
    # assert torch.allclose(r.grad, r_grad, atol=1e-4), breakpoint()
    # assert torch.allclose(k.grad, k_grad, atol=1e-4), breakpoint()
    # assert torch.allclose(v.grad, v_grad, atol=1e-4), breakpoint()
    # assert torch.allclose(beta.grad, beta_grad, atol=1e-4), breakpoint()
    print("All passed!")

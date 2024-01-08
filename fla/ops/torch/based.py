# -*- coding: utf-8 -*-

import torch
from einops import rearrange


def torch_parallel_based(q, k, v):
    q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn = 1 + attn + 1/2 * (attn ** 2)
    attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    return o


def torch_chunk_based(q, k, v, chunk_size=256):
    # constant term
    _o = v.cumsum(-2)

    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    q = q * (q.shape[-1] ** -0.5)

    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)

    intra_chunk_attn = q @ k.transpose(-2, -1)
    intra_chunk_attn = intra_chunk_attn + 1/2 * (intra_chunk_attn ** 2)
    intra_chunk_attn.masked_fill_(
        ~torch.tril(
            torch.ones(chunk_size, chunk_size,
                       dtype=torch.bool, device=q.device),
        ), 0)
    o = intra_chunk_attn @ v

    # quadractic term
    kv = torch.einsum(
        'b h n c x, b h n c y, b h n c z -> b h n x y z', k, k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)

    o += 0.5 * torch.einsum('b h n x y z, b h n c x, b h n c y -> b h n c z', kv, q, q)

    # linear term
    kv = torch.einsum('b h n c x, b h n c y -> b h n x y', k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    o += torch.einsum('b h n x y, b h n c x -> b h n c y', kv, q)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    return o + _o


if __name__ == "__main__":
    B = 4
    H = 4
    L = 256
    D = 16
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, D).cuda().to(dtype).requires_grad_(True)

    do = torch.randn_like(v).cuda() / 10
    ref = torch_parallel_based(q, k, v)
    ref.backward(do, retain_graph=True)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = torch_chunk_based(q, k, v)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-4), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-4), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-4), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-4), breakpoint()

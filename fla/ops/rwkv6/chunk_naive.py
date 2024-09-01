# -*- coding: utf-8 -*-

import torch
from einops import rearrange

from fla.ops.rwkv6.chunk import chunk_rwkv6
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6


def naive_chunk_rwkv6(
    q,
    k,
    v,
    w,
    u,
    chunk_size=32,
    initial_state=None,
    output_final_state=True,
):
    assert q.shape[-2] % chunk_size == 0
    orig_dtype = q.dtype
    num_chunk = q.shape[-2] // chunk_size
    if u.dim() == 2:
        u = torch.broadcast_to(u.unsqueeze(0), (q.shape[0], *u.shape))

    q, k, v, w = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size).float(), (q, k, v, w))

    w_cumsum = w.cumsum(-2)

    kw = k * (w_cumsum[..., -1, None, :] - w_cumsum).exp()
    wkv = kw.transpose(-1, -2) @ v

    wkv_new = torch.zeros_like(wkv)

    for i in range(num_chunk - 1):
        wkv_new[:, :, i + 1] = (wkv_new[:, :, i] * w_cumsum[:, :, i, -1, :, None].exp()) + wkv[:, :, i]

    o_inter = torch.einsum('b h n d p, b h n c d -> b h n c p', wkv_new, (q * (w_cumsum - w).exp()))

    o_intra = torch.zeros_like(o_inter)
    for i in range(chunk_size):
        attn = (q[:, :, :, i, None] * k * (w_cumsum[:, :, :, i, None] - w[:, :, :, i, None] - w_cumsum).exp()).sum(-1)
        mask = (torch.arange(0, chunk_size) < i).to(attn.device)
        attn.masked_fill_(~mask, 0)
        intra_inter_o = (attn.unsqueeze(-1) * v).sum(-2)
        intra_intra_o = (q[:, :, :, i] * u.unsqueeze(2) * k[:, :, :, i]).sum(-1).unsqueeze(-1) * v[:, :, :, i]
        o_intra[:, :, :, i] = intra_inter_o + intra_intra_o
    o = o_inter + o_intra
    return rearrange(o, 'b h n c d -> b h (n c) d').to(orig_dtype)


if __name__ == "__main__":
    from fla.utils import get_available_device
    from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6
    device = get_available_device()
    B = 4
    H = 32
    L = 1024
    D = 64
    dtype = torch.bfloat16
    require_grad = True
    torch.manual_seed(42)
    q = (torch.randn(B, H, L, D).uniform_(-1, 1).to(device).to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).uniform_(-1, 1).to(device).to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, D).uniform_(-1, 1).to(device).to(dtype).requires_grad_(require_grad)
    w = torch.randn(B, H, L, D).uniform_(-8, 1).to(device).to(dtype).requires_grad_(require_grad)
    u = (torch.randn(B, H, D).uniform_(-1, 1).to(device).to(dtype)).requires_grad_(require_grad)
    h = torch.randn(B, H, D, D, device=device, dtype=dtype, requires_grad=True)
    do = torch.rand_like(v).to(device)

    o, _ = fused_recurrent_rwkv6(q, k, v, w, u, initial_state=h, scale=1.0)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    o2, _ = chunk_rwkv6(q, k, v, w, u, initial_state=h, scale=1.0)
    o2.backward(do)

    def rmsre(pred, target, eps=1e-8):
        return torch.sqrt(torch.mean(torch.square((pred - target) / (target.abs() + eps))))

    def print_diff(name, grad1, grad2):
        abs_diff = (grad1 - grad2).abs()
        max_diff = abs_diff.max().item()
        rmsre_value = rmsre(grad1, grad2).item()
        print(f"{name}: Max Abs Diff = {max_diff:.6f}, RMSRE = {rmsre_value:.6f}")

    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(u.grad).any()
    assert not torch.isnan(h.grad).any()

    print(f"o: {(o - o2).abs().max().item():.6f}")
    print_diff("q", q.grad, dq)
    print_diff("k", k.grad, dk)
    print_diff("v", v.grad, dv)
    print_diff("w", w.grad, dw)
    print_diff("u", u.grad, du)
    print_diff("h", h.grad, dh)

    all_grads1 = torch.cat([q.grad.flatten(), k.grad.flatten(), v.grad.flatten(),
                            w.grad.flatten(), u.grad.flatten(), h.grad.flatten()])
    all_grads2 = torch.cat([dq.flatten(), dk.flatten(), dv.flatten(),
                            dw.flatten(), du.flatten(), dh.flatten()])
    overall_rmsre = rmsre(all_grads1, all_grads2).item()
    print(f"\nOverall RMSRE: {overall_rmsre:.6f}")

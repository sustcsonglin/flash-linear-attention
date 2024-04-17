# -*- coding: utf-8 -*-

import torch

from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6


def naive_recurrent_rwkv6(
    q,
    k,
    v,
    w,
    u,
    initial_state=None,
    output_final_state=False
):
    orig_dtype = q.dtype
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)


def naive_recurrent_rwkv6_bwd(
    q,
    k,
    v,
    w,
    u,
    o,
    do,
    initial_state=None,
    output_final_state=False
):
    q, k, v, w, u, o, do = map(lambda x: x.float(), (q, k, v, w, u, o, do))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h_i = (h + u[None, ..., None] * kv_i)
        dq_i = (do[:, :, i, None, :] * h_i).sum(-1)
        dq_aux_i = (do[:, :, i, None, :] * h).sum(-1)
        dq[:, :, i] = dq_i
        dq_aux[:, :, i] = dq_aux_i
        h = h * w_i[..., None] + kv_i

    du = torch.zeros_like(u)
    dh = torch.zeros_like(h)
    dk = torch.zeros_like(k)
    dk_aux = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for i in range(seq_len-1, -1, -1):
        d_kv_i = do[:, :, i, None, :] * q[:, :, i, :, None]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        du_i = (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(-1)
        du += du_i
        dk_i = (dh * v_i[..., None, :]).sum(-1)
        dk_aux[:, :, i] = dk_i
        dk_i += (d_kv_i * u[None, ..., None] * v_i[..., None, :]).sum(-1)
        dv_i = (d_kv_i * u[None, ..., None] * k_i[..., None]).sum(-2)
        dv_i += (dh * k_i[..., None]).sum(-2)

        dk[:, :, i] = dk_i
        dv[:, :, i] = dv_i
        dh = dh * w[:, :, i, :, None].exp() + d_kv_i

    # dw = q * dq_aux - k * dk_aux
    dw = torch.zeros_like(w)
    for i in range(seq_len-2, -1, -1):
        dw[:, :, i] = dw[:, :, i+1] + dq_aux[:, :, i+1] * q[:, :, i+1] - dk_aux[:, :, i] * k[:, :, i]

    return dq, dk, dv, dw, du


if __name__ == "__main__":
    B = 4
    H = 4
    L = 1024
    D = 100
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, 2*D).cuda().to(dtype).requires_grad_(True)
    w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).cuda().to(torch.float32).requires_grad_(True)
    u = (torch.randn(H, D).cuda().to(dtype)).requires_grad_(True)
    do = torch.rand_like(v).cuda()
    o = naive_recurrent_rwkv6(q, k, v, w, u)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    o2, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0)
    o2.backward(do)
    assert o.allclose(o2, 0, 1e-4), breakpoint()
    assert q.grad.allclose(dq, 0, 1e-4), breakpoint()
    assert k.grad.allclose(dk, 0, 1e-4), breakpoint()
    assert v.grad.allclose(dv, 0, 1e-4), breakpoint()
    assert u.grad.allclose(du, 0, 1e-3), breakpoint()
    assert w.grad.allclose(dw, 0, 1e-4), breakpoint()
    print("All tests passed!")
    # dq, dk, dv, dw, du = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do)
    # assert q.grad.allclose(dq, 0, 1e-5), breakpoint()
    # assert k.grad.allclose(dk, 0, 1e-5), breakpoint()
    # assert v.grad.allclose(dv, 0, 1e-5), breakpoint()
    # assert w.grad.allclose(dw, 0, 1e-5), breakpoint()
    # assert u.grad.allclose(du, 0, 1e-5), breakpoint()
    # breakpoint()

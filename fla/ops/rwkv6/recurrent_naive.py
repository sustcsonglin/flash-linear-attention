# -*- coding: utf-8 -*-

from typing import Optional

import torch


def naive_recurrent_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if scale is None:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht

@torch.no_grad
@torch.jit.script
def naive_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
):
    q, k, v, w, u, o, do = (x.to(dtype=torch.float32) for x in (q, k, v, w, u, o, do))
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    for i in range(T):
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

    for i in range(T - 1, -1, -1):
        d_kv_i = do[:, :, i, None, :] * q[:, :, i, :, None]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        du_i = (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(-1)
        du += du_i.sum(0)
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
    for i in range(T - 2, -1, -1):
        dw[:, :, i] = dw[:, :, i+1] + dq_aux[:, :, i+1] * q[:, :, i+1] - dk_aux[:, :, i] * k[:, :, i]

    return dq, dk, dv, dw, du, dh

if __name__ == "__main__":
    device = "cuda"
    B = 4
    H = 4
    L = 1024
    D = 100
    dtype = torch.float
    require_grad = True
    q = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, 2*D).to(device).to(dtype).requires_grad_(require_grad)
    w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).to(device).to(dtype).requires_grad_(require_grad)
    u = (torch.randn(H, D).to(device).to(dtype)).requires_grad_(require_grad)
    do = torch.rand_like(v).to(device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=torch.float32, requires_grad=True)
    o, _ = naive_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    dq2, dk2, dv2, dw2, du2, dh2 = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do, initial_state=h)

    assert torch.allclose(dq, dq2, atol=1e-3)
    assert torch.allclose(dk, dk2, atol=1e-3)
    assert torch.allclose(dv, dv2, atol=1e-3)
    assert torch.allclose(dw, dw2, atol=1e-3)
    assert torch.allclose(du, du2, atol=1e-3)
    assert torch.allclose(dh, dh2, atol=1e-3)
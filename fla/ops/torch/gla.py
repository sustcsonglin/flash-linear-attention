import chunk

import torch
import torch.nn.functional as F
from fla.ops.triton.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


def ceildiv(a, b):
    return -(a // -b)


def naive_loop(q, k, v, gk, initial_state=None, stop_grad=False):
    orig_dtype = q.dtype
    q, k, v, gk = map(lambda x: x.float(), (q, k, v, gk))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v,
                    dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)
    scale = d_head_k ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o_i = (q_i[..., None] * h).sum(-2)
        o[:, :, i] = o_i

    return o.to(orig_dtype), h


def naive_chunk_gla(q, k, v, gk, stop_grad=False):
    orig_dtype = q.dtype
    q, k, v, gk = map(lambda x: x.float(), (q, k, v, gk))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    chunk_size = 16
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v,
                    dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)
    scale = d_head_k ** -0.5

    for i in range(0, ceildiv(seq_len, chunk_size)):
        lo = i * chunk_size
        up = min((i + 1) * chunk_size, seq_len)
        q_chunk = q[:, :, lo:up, :] * scale
        k_chunk = k[:, :, lo:up]
        v_chunk = v[:, :, lo:up, :]
        gk_chunk = (gk[:, :, lo:up]).cumsum(-2)
        q_chunk = (q_chunk * gk_chunk.exp())
        k_chunk1 = (k_chunk * (-gk_chunk+gk_chunk[:, :, -1, None]).exp())
        kv_chunk = k_chunk1.transpose(-2, -1) @ v_chunk
        qk = q_chunk @ (k_chunk * (-gk_chunk).exp()).transpose(-2, -1)
        mask = torch.tril(torch.ones(
            q_chunk.shape[-2], q_chunk.shape[-2], dtype=torch.bool, device=q.device))
        qk = qk.masked_fill(~mask, 0)
        o_chunk = qk @ v_chunk
        o_chunk += q_chunk @ h
        o[:, :, lo:up, :] = o_chunk
        # if stop_grad:
        #     with torch.no_grad():
        #         decay = gk_chunk[:, :, -1].clone().detach().exp()
        # else:
        decay = gk_chunk[:, :, -1].exp()
        h = h * decay[..., None] + kv_chunk

    return o.to(orig_dtype)


if __name__ == "__main__":
    B = 4
    H = 4
    L = 512
    D = 128
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, D).cuda().to(dtype).requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, H, L, D)).cuda(
    ).clamp_min(-1).to(torch.float32).requires_grad_(True)

    do = torch.rand_like(v).cuda()
    intial_state = torch.randn(B, H, D, D).cuda()

    ref, ref_state = naive_loop(
        q, k, v, g, initial_state=intial_state)

    ref.backward(do, retain_graph=True)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, tri_state = fused_recurrent_gla(
        q, k, v, g, scale=D**-0.5, initial_state=intial_state, output_final_state=True)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-5), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()
    assert ref_dg.allclose(tri_dg, 0, 1e-4), breakpoint()

    tri, tri_state = fused_chunk_gla(
        q, k, v, g, scale=D**-0.5, initial_state=intial_state, output_final_state=True)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-5), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()
    assert ref_dg.allclose(tri_dg, 0, 1e-4), breakpoint()
    breakpoint()
    print("Pass")

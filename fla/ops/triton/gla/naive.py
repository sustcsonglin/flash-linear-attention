from sys import orig_argv
import torch
import torch.nn.functional as F
# from fla.ops.triton.gla.chunk_fuse import fused_chunk_gla
from fla.ops.triton.gla.recurrent_fuse import fused_recurrent_gla


def ceildiv(a, b):
    return -(a // -b)

def naive_loop(q, k, v, gk, stop_grad=False):
    orig_dtype = q.dtype
    q, k, v, gk = map(lambda x: x.float(), (q, k, v, gk))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v,
                    dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)
    scale = d_head_k ** -0.5

    for i in range(seq_len):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o_i = (q_i[..., None] * h).sum(-2)
        o[:, :, i] =  o_i 

    return o.to(orig_dtype)


if __name__ == "__main__":
    B = 4
    H = 4
    L = 32
    D = 64
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, D).cuda().to(dtype).requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, H, L, D)).cuda(
    ).clamp_min(-3).to(dtype).requires_grad_(True)

    do = torch.randn_like(v).cuda() / 10
    ref = naive_loop(q, k, v, g, stop_grad=False)
    ref.backward(do, retain_graph=True)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    no_grad = naive_loop(q, k, v, g, stop_grad=True)
    no_grad.backward(do, retain_graph=True)
    no_grad_dq, q.grad = q.grad.clone(), None
    no_grad_dk, k.grad = k.grad.clone(), None
    no_grad_dv, v.grad = v.grad.clone(), None
    no_grad_dg, g.grad = g.grad.clone(), None

    # assert ref.allclose(no_grad, 0, 1e-3), breakpoint()
    # assert ref_dq.allclose(no_grad_dq, 0, 1e-3), breakpoint()
    # assert ref_dk.allclose(no_grad_dk, 0, 1e-3), breakpoint()
    # assert ref_dv.allclose(no_grad_dv, 0, 1e-3), breakpoint()
    # assert ref_dg.allclose(no_grad_dg, 0, 1e-2), breakpoint()
    # g.grad.zero_()
    tri = fused_recurrent_gla(q, k, v, g)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-2), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-2), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-2), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-2), breakpoint()
    assert ref_dg.allclose(tri_dg, 0, 1e-2), breakpoint()

# -*- coding: utf-8 -*-

from audioop import reverse

import torch
import torch.nn.functional as F
from fla.ops.triton.retention import chunk_retention, fused_chunk_retention


def ceildiv(a, b):
    return -(a // -b)


def test_chunk():
    B = 4
    H = 4
    L = 324
    D = 123
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, 2 * D).cuda().to(dtype).requires_grad_(True)

    do = torch.rand_like(v).cuda()

    ref = chunk_retention(q, k, v)

    ref.backward(do, retain_graph=True)
    # ref_rev.backward(do2, retain_graph=True)

    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = fused_chunk_retention(
        q, k, v)
    tri.backward(do, retain_graph=True)
    # tri_rev.backward(do2, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-5), breakpoint()
    # assert ref_rev.allclose(tri_rev, 0, 1e-5), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()

    # tri = fused_chunk_gla(
    #     q, k, v, g)
    # tri.backward(do, retain_graph=True)
    # tri_dq, q.grad = q.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dg, g.grad = g.grad.clone(), None

    # assert ref.allclose(tri, 0, 1e-5), breakpoint()
    # assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    # assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    # assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()
    # assert ref_dg.allclose(tri_dg, 0, 1e-4), breakpoint()
    # breakpoint()
    breakpoint()
    print("Pass")


if __name__ == "__main__":
    test_chunk()
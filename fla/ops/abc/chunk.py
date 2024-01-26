# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.utils import contiguous


@triton.jit
def safe_exy(x, y):
    # e^x * y = sign(y) * e^(x + log(|y|)
    return tl.where(y > 0, 1., -1.) * tl.exp(x + tl.log(tl.abs(y.to(tl.float32))))


@triton.jit
def chunk_abc_fwd_kernel_cum(
    s,
    r,
    c,
    p,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * M,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    b_mp = tl.zeros([BM,], dtype=tl.float32)
    b_zp = tl.zeros([BM,], dtype=tl.float32)
    for i in range(NT):
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

        b_m = tl.max(b_s, 0)
        # workaround for compiler bugs
        if i == 0:
            b_r = tl.full([BM,], 1., dtype=tl.float32)
        else:
            b_m = tl.maximum(b_mp, b_m)
            b_r = tl.exp(b_mp - b_m)
        b_c = tl.exp(b_s - b_m[None, :])
        b_z = tl.cumsum(b_c, 0) + (b_zp * b_r)[None, :]
        b_p = -tl.log(b_z)
        b_mp = b_m
        b_zp = tl.max(b_z, 0)

        tl.store(p_r, b_r.to(p_r.dtype.element_ty), boundary_check=(0,))
        tl.store(p_c, b_c.to(p_c.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))

        p_s = tl.advance(p_s, (BT, 0))
        p_r = tl.advance(p_r, (M,))
        p_c = tl.advance(p_c, (BT, 0))
        p_p = tl.advance(p_p, (BT, 0))


@triton.jit
def chunk_abc_fwd_kernel_h(
    k,
    v,
    r,
    h,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMQ: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, ((i+1)*K, V), (s_h_t, s_h_d), (i*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        if NORMQ:
            p_r = tl.make_block_ptr(r + i_bh * s_k_t * NT, ((i+1)*K,), (s_k_d,), (i*K+i_k*BK,), (BK,), (0,))
            b_r = tl.load(p_r, boundary_check=(0,))
            b_h = b_h * b_r[:, None]
        else:
            p_r = tl.make_block_ptr(r + i_bh * s_v_t * NT, ((i+1)*V,), (s_v_d,), (i*V+i_v*BV,), (BV,), (0,))
            b_r = tl.load(p_r, boundary_check=(0,))
            b_h = b_h * b_r[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)


@triton.jit
def chunk_abc_fwd_kernel_o(
    q,
    k,
    v,
    h,
    p,
    o,
    scale,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    NORMQ: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (0, i_t * BT), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, ((i_t+1)*K, V), (s_h_t, s_h_d), (i_t * K, i_v * BV), (BK, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if SCALE:
            b_q = (b_q * scale).to(b_q.dtype)
        if NORMQ:
            p_p = tl.make_block_ptr(p + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_p = tl.load(p_p, boundary_check=(0, 1))
            b_q = safe_exy(b_p, b_q).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        # [BT, BT]
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

        p_q = tl.advance(p_q, (0, BK))
        p_k = tl.advance(p_k, (BK, 0))
        p_h = tl.advance(p_h, (BK, 0))
    b_s = tl.where(m_s, b_s, 0.)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
    if not NORMQ:
        p_p = tl.make_block_ptr(p + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_o = safe_exy(b_p, b_o)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dh(
    q,
    p,
    r,
    do,
    dh,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    SCALE: tl.constexpr,
    NORMQ: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, ((i+1)*K, V), (s_h_t, s_h_d), (i*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if SCALE:
            b_do = (b_do * scale).to(b_do.dtype)
        if NORMQ:
            p_p = tl.make_block_ptr(p + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
            p_r = tl.make_block_ptr(r + i_bh * s_k_t * NT, (NT*K,), (s_k_d,), (((i+1) % NT)*K+i_k*BK,), (BK,), (0,))
            # [BK, BT]
            b_p = tl.load(p_p, boundary_check=(0, 1))
            # [BK]
            b_r = tl.load(p_r, boundary_check=(0,))
            # [BK, BT]
            b_q = safe_exy(b_p, b_q).to(b_q.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[:, None]
        else:
            p_p = tl.make_block_ptr(p + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
            p_r = tl.make_block_ptr(r + i_bh * s_v_t * NT, (NT*V,), (s_v_d,), (((i+1) % NT)*V+i_v*BV,), (BV,), (0,))
            # [BT, BV]
            b_p = tl.load(p_p, boundary_check=(0, 1))
            # [BV]
            b_r = tl.load(p_r, boundary_check=(0,))
            # [BT, BV]
            b_do = safe_exy(b_p, b_do).to(b_do.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[None, :]
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_abc_bwd_kernel_dqkv(
    q,
    k,
    v,
    h,
    p,
    do,
    dh,
    dq,
    dk,
    dv,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    NORMQ: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    m_s, m_t = o_i[:, None] >= o_i[None, :], o_i[:, None] <= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, (i_t+1)*K), (s_h_d, s_h_t), (0, i_t * K + i_k * BK), (BV, BK), (0, 1))

    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, ((i_t+1)*K, V), (s_h_t, s_h_d), (i_t * K + i_k * BK, 0), (BK, BV), (1, 0))

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))

    # [BK, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # still remains potential stability issues
    if NORMQ:
        p_p = tl.make_block_ptr(p + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_q = safe_exy(b_p, b_q).to(b_q.dtype)
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_s = tl.where(m_t, tl.dot(b_k, b_q, allow_tf32=False), 0).to(b_q.dtype)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        if SCALE:
            b_do = (b_do * scale).to(b_do.dtype)
        if not NORMQ:
            p_p = tl.make_block_ptr(p + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_p = tl.load(p_p, boundary_check=(0, 1))
            b_do = safe_exy(b_p, b_do).to(b_do.dtype)
        # [BT, BT]
        b_ds = tl.where(m_s, tl.dot(b_do, tl.trans(b_v), allow_tf32=False), 0).to(b_v.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) + tl.dot(b_ds, b_k, allow_tf32=False)

        # [BT, BT]
        b_ds = tl.trans(b_ds)
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) + tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)

        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_s, b_do, allow_tf32=False)
        if not NORMQ:
            b_dv = b_v * b_dv
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.advance(p_v, (0, BV))
        p_h = tl.advance(p_h, (BV, 0))
        p_do = tl.advance(p_do, (0, BV))
        p_dh = tl.advance(p_dh, (0, BV))
        p_dv = tl.advance(p_dv, (0, BV))

    if NORMQ:
        p_p = tl.make_block_ptr(p + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_dq = safe_exy(b_p, b_dq)
        b_dk = b_k * b_dk

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_rcum(
    s,
    r,
    c,
    p,
    o,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_t = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BM,], dtype=tl.float32)
    for i in range(NT - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_m), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * M,), (s_sk_m,), (((i + 1) % NT) * M + i_m * BM,), (BM,), (0,))
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = tl.load(p_c, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        # [BM,]
        b_r = tl.load(p_r, boundary_check=(0,))

        b_z = b_z * b_r
        # [BT, BM]
        b_s = safe_exy(b_p, b_s.to(tl.float32)).to(b_s.dtype)
        b_o -= b_c * (b_z[None, :] + tl.dot(m_t.to(b_s.dtype), b_s, allow_tf32=False))

        # [BM,]
        b_z += tl.sum(b_s, 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class ChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        B, H, T, K, V, M = *q.shape, v.shape[-1], sk.shape[-1]
        BT = 64
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NM = triton.cdiv(T, BT), triton.cdiv(M, BM)
        scale = K ** -0.5
        num_stages = 1
        num_warps = 2 if BK == 64 else 1

        def fwd_inner(q, k, v, r, p, T, K, V, BT, BK, BV, NT, scale=None, normq=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_abc_fwd_kernel_h[grid](
                k, v, r, h,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                NORMQ=normq,
                num_warps=num_warps,
                num_stages=num_stages
            )
            o = q.new_empty(B, H, T, V)
            grid = (NV, NT, B * H)
            chunk_abc_fwd_kernel_o[grid](
                q, k, v, h, p, o, scale,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
                SCALE=(scale is not None),
                NORMQ=normq,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return o, h

        rk, ck, pk = sk.new_empty(B, H, NT, M), torch.empty_like(sk), torch.empty_like(sk)
        grid = (NM, B * H)
        chunk_abc_fwd_kernel_cum[grid](
            sk, rk, ck, pk,
            sk.stride(1), sk.stride(2), sk.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        s, hk = fwd_inner(
            q=q,
            k=k,
            v=ck,
            r=rk,
            p=pk,
            T=T,
            K=K,
            V=M,
            BT=BT,
            BK=BK,
            BV=BM,
            NT=NT,
            scale=scale,
            normq=False
        )
        p = s.softmax(-1, dtype=torch.float).to(q.dtype)
        rv, cv, pv = sv.new_empty(B, H, NT, M), torch.empty_like(sv), torch.empty_like(sv)
        grid = (NM, B * H)
        chunk_abc_fwd_kernel_cum[grid](
            sv, rv, cv, pv,
            sv.stride(1), sv.stride(2), sv.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        o, hv = fwd_inner(
            q=p,
            k=cv,
            v=v,
            r=rv,
            p=pv,
            T=T,
            K=M,
            V=V,
            BT=BT,
            BK=BM,
            BV=BV,
            NT=NT,
            scale=None,
            normq=True
        )
        ctx.save_for_backward(q, k, v, o, s, p, rk, ck, pk, hk, rv, cv, pv, hv)
        ctx.BT = BT
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, p, rk, ck, pk, hk, rv, cv, pv, hv = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT = ctx.BT
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NM = triton.cdiv(T, BT), triton.cdiv(M, BM)
        scale = K ** -0.5
        num_warps = 2 if BK == 64 else 1
        num_stages = 1

        def bwd_inner(q, k, v, h, r, p, do, T, K, V, BT, BK, BV, NT, scale=None, normq=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = torch.empty_like(h)
            num_warps = 2 if BK == 64 else 1
            grid = (NK, NV, B * H)
            chunk_abc_bwd_kernel_dh[grid](
                q, p, r, do, dh,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                SCALE=(scale is not None),
                NORMQ=normq,
                num_warps=num_warps,
                num_stages=num_stages
            )

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = v.new_empty(NK, *v.shape)
            num_warps = 4 if BK == 64 else 2
            grid = (NK, NT, B * H)
            chunk_abc_bwd_kernel_dqkv[grid](
                q, k, v, h, p, do, dh, dq, dk, dv,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
                SCALE=(scale is not None),
                NORMQ=normq,
                num_warps=num_warps,
                num_stages=num_stages
            )
            dv = dv.sum(0)
            return dq, dk, dv

        dp, dsv, dv = bwd_inner(
            q=p,
            k=cv,
            v=v,
            h=hv,
            r=rv,
            p=pv,
            do=do,
            T=T,
            K=M,
            V=V,
            BT=BT,
            BK=BM,
            BV=BV,
            NT=NT,
            scale=None,
            normq=True
        )
        # grad of softmax
        ds = p * (dp - (o * do).sum(-1, True))
        dq, dk, dsk = bwd_inner(
            q=q,
            k=k,
            v=ck,
            h=hk,
            r=rk,
            p=pk,
            do=ds,
            T=T,
            K=K,
            V=M,
            BT=BT,
            BK=BK,
            BV=BM,
            NT=NT,
            scale=scale,
            normq=False
        )
        grid = (NM, B * H)
        chunk_abc_bwd_kernel_rcum[grid](
            ds * s, rk, ck, pk, dsk,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        chunk_abc_bwd_kernel_rcum[grid](
            p * dp, rv, cv, pv, dsv,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq, dk, dv, dsk, dsv


chunk_abc = ChunkABCFunction.apply

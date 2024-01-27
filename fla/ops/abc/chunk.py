# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.utils import contiguous


@triton.jit
def safe_exy(x, y):
    # e^x * y = sign(y) * e^(x + log(|y|)
    # this utility is designed for the safe multiplication of two variables e^x and y,
    # where x is in log space and y in normal space respectively.
    # it is important to ensure that e^x * y will not result in an overflow
    return tl.where(y > 0, 1., -1.) * tl.exp(x + tl.log(tl.abs(y.to(tl.float32))))


@triton.jit
def chunk_abc_fwd_kernel_m(
    s,
    r,
    s_sk_h,
    s_sk_t,
    s_sk_d,
    T: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NT: tl.constexpr,
    NM: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_r = tl.make_block_ptr(r + i_bh * NT, (NT,), (s_sk_d,), (i_t,), (1,), (0,))
    b_m = tl.full([1,], float('-inf'), dtype=tl.float32)
    for i_m in range(NM):
        p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i_t * BT, i_m * BM), (BT, BM), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_m = tl.maximum(tl.max(tl.max(b_s, 0), 0), b_m)
    tl.store(p_r, b_m.to(p_r.dtype.element_ty), boundary_check=(0,))


@triton.jit
def chunk_abc_fwd_kernel_cum(
    s,
    r,
    c,
    z,
    s_sk_h,
    s_sk_t,
    s_sk_d,
    T: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)

    b_mp = tl.full([1,], float('-inf'), dtype=tl.float32)
    b_sp = tl.zeros([BM,], dtype=tl.float32)
    for i_t in range(NT):
        p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i_t * BT, i_m * BM), (BT, BM), (1, 0))
        p_r = tl.make_block_ptr(r + i_bh * NT, (NT,), (s_sk_d,), (i_t,), (1,), (0,))
        p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i_t * BT, i_m * BM), (BT, BM), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i_t * BT, i_m * BM), (BT, BM), (1, 0))

        b_m = tl.load(p_r, boundary_check=(0,)).to(tl.float32)
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

        # workaround for compiler bugs
        if i_t == 0:
            b_r = tl.zeros([1,], dtype=tl.float32)
        else:
            b_m = tl.maximum(b_mp, b_m)
            b_r = (b_mp - b_m).to(tl.float32)
        b_c = tl.exp(b_s - b_m)
        b_s = tl.cumsum(b_c, 0) + (b_sp * tl.exp(b_r))[None, :]
        b_z = tl.exp(b_r - tl.log(b_s))
        b_sp = tl.max(b_s, 0)

        tl.store(p_r, b_r.to(p_r.dtype.element_ty), boundary_check=(0,))
        tl.store(p_c, b_c.to(p_c.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))

        b_mp = b_m


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
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, ((i_t+1)*K, V), (s_h_t, s_h_d), (i_t*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        # crashed if no condition checks
        if i_t > 0:
            b_r = tl.load(r + i_bh * NT + i_t).to(tl.float32)
            b_h = b_h * tl.exp(b_r)
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
    r,
    h,
    z,
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
    NT: tl.constexpr,
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
        if NORMQ:
            p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_q = b_z * b_q
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
    if SCALE:
        b_o = b_o * scale
    if not NORMQ:
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_o = b_z * b_o
    p_r = tl.make_block_ptr(r + i_bh * NT, (NT,), (s_k_d,), (i_t,), (1,), (0,))
    b_r = tl.load(p_r, boundary_check=(0,)).to(tl.float32)
    b_o = safe_exy(-b_r, b_o)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dh(
    q,
    z,
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
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, ((i_t+1)*K, V), (s_h_t, s_h_d), (i_t*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if SCALE:
            b_do = (b_do * scale).to(b_do.dtype)
        if NORMQ:
            p_z = tl.make_block_ptr(z + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            # [BK, BT]
            b_z = tl.load(p_z, boundary_check=(0, 1))
            # [BK, BT]
            b_q = (b_z * b_q).to(b_q.dtype)
        else:
            p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            # [BT, BV]
            b_z = tl.load(p_z, boundary_check=(0, 1))
            # [BT, BV]
            b_do = (b_z * b_do).to(b_do.dtype)
        if i_t < NT - 1:
            p_r = tl.make_block_ptr(r + i_bh * NT, (NT,), (1,), (i_t,), (1,), (0,))
            b_r = tl.load(p_r, boundary_check=(0,)).to(tl.float32)
            # [BK, BV]
            b_dh = b_dh * tl.exp(b_r)
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_abc_bwd_kernel_dqkv(
    q,
    k,
    v,
    r,
    h,
    z,
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
    NT: tl.constexpr,
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

    b_r = tl.load(r + i_bh * NT + i_t).to(tl.float32)
    # [BK, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    if NORMQ:
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_q = b_z * b_q
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
            p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_do = b_z * b_do
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
        b_dv = safe_exy(-b_r, b_dv)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.advance(p_v, (0, BV))
        p_h = tl.advance(p_h, (BV, 0))
        p_do = tl.advance(p_do, (0, BV))
        p_dh = tl.advance(p_dh, (0, BV))
        p_dv = tl.advance(p_dv, (0, BV))

    if NORMQ:
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_dq = b_z * b_dq
        b_dk = b_k * b_dk
    b_dq = safe_exy(-b_r, b_dq)
    b_dk = safe_exy(-b_r, b_dk)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_rcum(
    s,
    r,
    c,
    z,
    o,
    s_sk_h,
    s_sk_t,
    s_sk_d,
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

    b_sp = tl.zeros([BM,], dtype=tl.float32)
    for i in range(NT - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i * BT, i_m * BM), (BT, BM), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_sk_h, (T, M), (s_sk_t, s_sk_d), (i * BT, i_m * BM), (BT, BM), (1, 0))
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = tl.load(p_c, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))

        b_r = tl.load(r + i_bh * NT + i).to(tl.float32)
        # [BM,]
        b_sp = b_sp * tl.exp(b_r)

        # [BT, BM]
        b_s = (b_z * b_s).to(b_s.dtype)
        b_o -= safe_exy(-b_r, b_c * (b_sp[None, :] + tl.dot(m_t.to(b_z.dtype), b_s, allow_tf32=False)))

        # [BM,]
        b_sp += tl.sum(b_s, 0)

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
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_inner(q, k, v, r, z, T, K, V, BT, BK, BV, NT, scale=None, normq=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_abc_fwd_kernel_h[grid](
                k, v, r, h,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages
            )
            o = torch.empty_like(v)
            grid = (NV, NT, B * H)
            chunk_abc_fwd_kernel_o[grid](
                q, k, v, r, h, z, o, scale,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                SCALE=(scale is not None),
                NORMQ=normq,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return o, h

        rk, ck, zk = sk.new_empty(B, H, NT), torch.empty_like(sk), torch.empty_like(sk)
        grid = (NT, B * H)
        chunk_abc_fwd_kernel_m[grid](
            sk, rk,
            sk.stride(1), sk.stride(2), sk.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT, NM=NM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NM, B * H)
        chunk_abc_fwd_kernel_cum[grid](
            sk, rk, ck, zk,
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
            z=zk,
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
        rv, cv, zv = sv.new_empty(B, H, NT), torch.empty_like(sv), torch.empty_like(sv)
        grid = (NT, B * H)
        chunk_abc_fwd_kernel_m[grid](
            sv, rv,
            sv.stride(1), sv.stride(2), sv.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT, NM=NM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NM, B * H)
        chunk_abc_fwd_kernel_cum[grid](
            sv, rv, cv, zv,
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
            z=zv,
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
        ctx.save_for_backward(q, k, v, o, s, p, rk, ck, zk, hk, rv, cv, zv, hv)
        ctx.BT = BT
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, p, rk, ck, zk, hk, rv, cv, zv, hv = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT = ctx.BT
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NM = triton.cdiv(T, BT), triton.cdiv(M, BM)
        scale = K ** -0.5
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def bwd_inner(q, k, v, h, r, z, do, T, K, V, BT, BK, BV, NT, scale=None, normq=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = torch.empty_like(h)
            grid = (NK, NV, B * H)
            chunk_abc_bwd_kernel_dh[grid](
                q, z, r, do, dh,
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
            grid = (NK, NT, B * H)
            chunk_abc_bwd_kernel_dqkv[grid](
                q, k, v, r, h, z, do, dh, dq, dk, dv,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
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
            z=zv,
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
            z=zk,
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
            ds * s, rk, ck, zk, dsk,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        chunk_abc_bwd_kernel_rcum[grid](
            p * dp, rv, cv, zv, dsv,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, M=M, BT=BT, BM=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq, dk, dv, dsk, dsv


chunk_abc = ChunkABCFunction.apply

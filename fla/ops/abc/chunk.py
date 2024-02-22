# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.utils import contiguous


@triton.jit
def logcumsumexp_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)

    b_mp = tl.full([BS,], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([BS,], dtype=tl.float32)
    for i_t in range(NT):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))

        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [BS,]
        b_mc = tl.max(b_s, 0)
        # workaround for compiler bugs
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        # [BT, BS]
        b_z = tl.cumsum(tl.exp(b_s - b_mc), 0) + b_zp
        # [BS,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-10)) + b_mc
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def softmax_fwd_kernel(
    s,
    p,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BT,]
    b_z = tl.zeros([BT,], dtype=tl.float32)
    b_m = tl.zeros([BT,], dtype=tl.float32)
    for i in range(tl.cdiv(S, BS)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        # [BT]
        b_mc = tl.max(b_s, 1)
        b_mc = tl.maximum(b_m, b_mc)
        if i > 0:
            b_z = b_z * tl.exp(b_m - b_mc)
        b_z += tl.sum(tl.exp(b_s - b_mc[:, None]), 1)
        b_m = b_mc
    b_z = tl.log(b_z) + b_m

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_p = tl.exp(b_s - b_z[:, None])
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def softmax_bwd_kernel(
    p,
    o,
    dp,
    do,
    ds,
    s_v_h,
    s_v_t,
    s_v_d,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BT,]
    b_doo = tl.zeros([BT,], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BT, BV]
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT,]
        b_doo += tl.sum(b_do * b_o, 1)
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_p = tl.load(p_p, boundary_check=(0, 1))
    b_dp = tl.load(p_dp, boundary_check=(0, 1))
    b_ds = b_p * (b_dp - b_doo[:, None])
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_h(
    k,
    v,
    z,
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
    NORMK: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.zeros([BK if NORMK else BV], dtype=tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if NORMK:
            p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            # [BK,]
            b_zc = tl.load(p_zc, boundary_check=(0,))
            if i_t == 0:
                b_zp = b_zc
            b_r, b_zp = tl.exp(b_zp - b_zc), b_zc
            # [BK, BV]
            b_h = b_h * b_r[:, None]
            b_k = tl.exp(b_k - b_zc[:, None]).to(b_k.dtype)
        else:
            p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            # [BV,]
            b_zc = tl.load(p_zc, boundary_check=(0,))
            if i_t == 0:
                b_zp = b_zc
            b_r, b_zp = tl.exp(b_zp - b_zc), b_zc
            # [BK, BV]
            b_h = b_h * b_r[None, :]
            b_v = tl.exp(b_v - b_zc[None, :]).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)


@triton.jit
def chunk_abc_fwd_kernel_intra_K(
    v,
    z,
    o,
    A,
    s_v_h,
    s_v_t,
    s_v_d,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    # [BV,]
    b_zn = tl.load(p_zn, boundary_check=(0,))
    # [BC, BV]
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, tl.exp(b_v - b_zn[None, :]).to(b_v.dtype), allow_tf32=False)
    b_o *= tl.exp(b_zn[None, :] - b_z)

    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
        # [BC, BV]
        # avoid 0 * inf = inf
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * tl.exp(b_v[None, :] - b_z), 0)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_K(
    q,
    k,
    h,
    z,
    o,
    A,
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
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k, allow_tf32=False)
    p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_zp = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT - 1) * V + i_v * BV,), (BV,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_z = tl.load(p_z, boundary_check=(0, 1))
    # [BV,]
    b_zp = tl.load(p_zp, boundary_check=(0,))
    # [BT, BV]
    if i_t > 0:
        b_o = b_o * tl.exp(b_zp[None, :] - b_z) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.) * scale
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_intra_V(
    q,
    k,
    z,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_zn = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
        # [BK,]
        b_zn = tl.load(p_zn, boundary_check=(0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_q = (b_q * tl.exp(b_zn[None, :] - b_z)).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = tl.exp(b_k - b_zn[:, None]).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))

        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BK,]
            b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
            # [BC,]
            b_A = tl.sum(b_q * tl.exp(b_k[None, :] - b_z), 1)
            b_A = tl.where(o_i >= j, b_A, 0)
            tl.store(A + o_A + j, b_A.to(b_q.dtype), mask=m_A)

            p_k = tl.advance(p_k, (K,))


@triton.jit
def chunk_abc_fwd_kernel_V(
    q,
    v,
    h,
    z,
    o,
    A,
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
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT - 1) * K + i_k * BK,), (BK,), (0,))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_z = tl.load(p_z, boundary_check=(0, 1))
        # [BT, BK]
        if i_t > 0:
            b_zp = tl.load(p_zp, boundary_check=(0,))
            b_q = (b_q * tl.exp(b_zp[None, :] - b_z)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dh(
    q,
    z,
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
    NORMK: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.full([BK if NORMK else BV], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        if SCALE:
            b_do = (b_do * scale).to(b_do.dtype)
        if NORMK:
            p_z = tl.make_block_ptr(z + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT - 1) * K + i_k * BK,), (BK,), (0,))
            # [BK,]
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            # [BK, BT]
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_q = (b_q * tl.exp(b_zc[:, None] - b_z)).to(b_q.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[:, None]
        else:
            p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT - 1) * V + i_v * BV,), (BV,), (0,))
            # [BV,]
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            # [BT, BV]
            b_z = tl.load(p_z, boundary_check=(0,))
            b_do = (b_do * tl.exp(b_zc[None, :] - b_z)).to(b_do.dtype)
            # [BK, BV]
            b_dh = b_dh * b_r[None, :]
        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_abc_bwd_kernel_V(
    k,
    v,
    h,
    z,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
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
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))

    # [BK,]
    b_zc = tl.load(p_zc, boundary_check=(0,))
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_k = tl.exp(b_k - b_zc[None, :]).to(b_k.dtype)
    # [BT, BT]
    if i_k == 0:
        b_A = tl.load(p_A, boundary_check=(0, 1))
    else:
        b_A = tl.zeros([BT, BT], dtype=b_k.dtype)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_zp = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT - 1) * K + i_k * BK,), (BK,), (0,))
    if i_t > 0:
        # [BK,]
        b_zp = tl.load(p_zp, boundary_check=(0,))
        # [BT, BK]
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_z = tl.exp(b_zp[None, :] - b_z)
        # [BT, BK]
        b_dq = b_dq * b_z
    b_dk = b_dk * b_k

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_V(
    q,
    k,
    z,
    dA,
    dq,
    dk,
    s_k_h,
    s_k_t,
    s_k_d,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
    # [BK,]
    b_zn = tl.load(p_zn, boundary_check=(0,))
    # [BC, BK]
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_zq = tl.exp(b_zn[None, :] - b_z)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_qz = (b_q * b_zq).to(b_q.dtype)
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk+(i_i*n_bh+i_bh)*s_k_h, (T, K), (s_k_t, s_k_d), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kz = tl.exp(b_k - b_zn[None, :]).to(b_k.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dq += tl.dot(b_dA, b_kz, allow_tf32=False)
        b_dk = tl.dot(tl.trans(b_dA), b_qz, allow_tf32=False) * b_kz
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    b_dq *= b_zq

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        p_dk = tl.make_block_ptr(dk+(i_i*n_bh+i_bh)*s_k_h, (T * K,), (1,), ((i_t*BT+i_i*BC + j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        b_kz = tl.where(m_i, tl.exp(b_k[None, :] - b_z), 0.)
        # [BC, BK]
        b_dq += b_dA[:, None] * b_kz
        # [BK,]
        b_dk = tl.sum(b_q * b_dA[:, None] * b_kz, 0)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,))
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_K(
    v,
    z,
    do,
    dA,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dA = tl.make_block_ptr(dA+(i_bh+i_v*n_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
        # [BV,]
        b_zn = tl.load(p_zn, boundary_check=(0,))
        # [BC, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_zn[None, :] - b_z)).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zn[:, None]).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_v, allow_tf32=False) * scale
        tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_j * BC) * V + i_v * BV,), (BV,), (0,))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))

        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_v * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BV,]
            b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
            # [BC,]
            b_dA = tl.sum(b_do * tl.exp(b_v[None, :] - b_z), axis=1)
            b_dA = tl.where(o_i >= j, b_dA, 0) * scale
            tl.store(dA + o_A + j, b_dA.to(b_do.dtype), mask=m_A)

            p_v = tl.advance(p_v, (V,))


@triton.jit
def chunk_abc_bwd_kernel_K(
    q,
    k,
    v,
    h,
    z,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
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
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh) * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.where(m_s, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0) * scale
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K*V, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT - 1) * V + i_v * BV,), (BV,), (0,))
        p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))

        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BV,]
        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zc[None, :]).to(b_v.dtype)
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_z = tl.exp(b_zp[None, :] - b_z)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * b_z * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BK]
        if i_t > 0:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = b_v * tl.dot(b_k, b_dh, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, allow_tf32=False)

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_KV(
    v,
    z,
    A,
    do,
    dv,
    s_v_h,
    s_v_t,
    s_v_d,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T*V,), (s_v_d,), ((i_t * BT + i_i * BC + BC - 1) * V + i_v * BV,), (BV,), (0,))
    # [BV,]
    b_zn = tl.load(p_zn, boundary_check=(0,))
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV),  (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_zn[None, :] - b_z)).to(b_do.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do, allow_tf32=False)
    b_dv *= tl.exp(b_v - b_zn[None, :])

    o_i = tl.arange(0, BC)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T * BT,), (1,), ((i_t * BT + i_i * BC + j) * BT + i_i * BC,), (BC,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        # [BC,]
        b_A = tl.load(p_A, boundary_check=(0,))
        # [BV,]
        b_z = tl.load(p_z, boundary_check=(0,))
        b_do = tl.load(p_do, boundary_check=(0,))
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_v - b_z[None, :]) * b_A[:, None] * b_do[None, :], 0.)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_rcum_inter(
    s,
    z,
    ss,
    doo,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)

    b_sp = tl.zeros([BS,], dtype=tl.float32)
    b_zp = tl.full([BS,], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_zc = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (s_s_d,), ((i_t * BT) * S + i_m * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        # [BS,]
        b_zc = tl.load(p_zc, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))

        b_doo = tl.exp(b_s - b_zp[None, :]) * b_sp[None, :]
        tl.store(p_doo, b_doo.to(p_doo.dtype.element_ty), boundary_check=(0, 1))
        # [BS,]
        b_sp = b_sp * tl.exp(b_zc - b_zp) + tl.sum(b_ss * tl.exp(b_zc[None, :] - b_z), 0)
        b_zp = b_zc


@triton.jit
def chunk_abc_bwd_kernel_rcum_intra(
    s,
    z,
    ss,
    doo,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr
):
    i_s, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    o_i = tl.arange(0, BC)
    m_o = tl.full([BC, BC], 1., dtype=tl.float32)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_s_h, (T*S,), (s_s_d,), ((i_t * BT + i_i * BC + BC - 1) * S + i_s * BS,), (BS,), (0,))
    p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    # [BC, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1))
    # [BS,]
    b_zn = tl.load(p_zn, boundary_check=(0,))

    b_doo = tl.zeros([BC, BS], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        # [BC, BS]
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        # [BC, BS]
        b_doo += b_ss * tl.exp(b_zn[None, :] - b_z)
    b_doo = tl.exp(b_s - b_zn[None, :]) * tl.dot(m_o.to(b_s.dtype), b_doo.to(b_s.dtype), allow_tf32=False)

    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        # [BS,]
        b_z = tl.load(p_z, boundary_check=(0,))
        b_ss = tl.load(p_ss, boundary_check=(0,))
        # [BC, BS]
        m_i = o_i[:, None] <= j
        b_doo += tl.where(m_i, tl.exp(b_s - b_z[None, :]) * b_ss[None, :], 0.)
    b_doo += tl.load(p_doo, boundary_check=(0, 1))
    tl.store(p_doo, b_doo.to(p_doo.dtype.element_ty), boundary_check=(0, 1))


class ChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, s):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NV, NM = triton.cdiv(V, BV), triton.cdiv(M, BM)
        scale = K ** -0.5
        num_warps = 4 if BK == 64 else 2
        num_stages = 1
        assert M % 64 == 0, "For efficiency, M must be a multiple of 64."

        def fwd_inner(q, k, v, z, B, H, T, K, V, BT, BK, BV, NT, normk=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_abc_fwd_kernel_h[grid](
                k, v, z, h,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                NORMK=normk,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return h

        # keep cummulative normalizer in fp32
        z = torch.empty_like(s, dtype=torch.float)
        grid = (NM, B * H)
        logcumsumexp_kernel[grid](
            s, z,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT, BS=BM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )

        hk = fwd_inner(
            q=q, k=k, v=s, z=z,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            normk=False
        )
        ok1 = torch.empty_like(s)
        Ak = q.new_empty(B, H, T, BT)
        grid = (NM, NT, B * H)
        chunk_abc_fwd_kernel_K[grid](
            q, k, hk, z, ok1, Ak,
            k.stride(1), k.stride(2), k.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            hk.stride(1), hk.stride(2), hk.stride(3),
            scale,
            T=T, K=K, V=M, BT=BT, BK=BK, BV=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ok0 = torch.empty_like(s)
        grid = (NM, NT * NC, B * H)
        chunk_abc_fwd_kernel_intra_K[grid](
            s, z, ok0, Ak,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages
        )
        ok = ok0.add_(ok1)

        p = torch.empty_like(ok)
        grid = (NM, NT, B * H)
        softmax_fwd_kernel[grid](
            ok, p,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT, BS=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        hv = fwd_inner(
            q=p, k=s, v=v, z=z,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
            normk=True
        )
        Av = q.new_zeros(NM, B, H, T, BT)
        grid = (NM, NT * NC * NC, B * H)
        chunk_abc_fwd_kernel_intra_V[grid](
            p, s, z, Av,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages
        )
        Av = Av.sum(0)
        ov = torch.empty_like(v)
        grid = (NV, NT, B * H)
        chunk_abc_fwd_kernel_V[grid](
            p, v, hv, z, ov, Av,
            p.stride(1), p.stride(2), p.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            hv.stride(1), hv.stride(2), hv.stride(3),
            T=T, K=M, V=V, BT=BT, BK=BM, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, s, z, ok, p, ov, hk, hv, Av)
        ctx.BT = BT
        return ov

    @staticmethod
    @contiguous
    def backward(ctx, dov):
        q, k, v, s, z, ok, p, ov, hk, hv, Av = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC, NK, NM = triton.cdiv(T, BT), triton.cdiv(BT, BC), triton.cdiv(K, BK), triton.cdiv(M, BM)
        scale = K ** -0.5
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def bwd_inner(q, z, do, B, H, T, K, V, BT, BK, BV, NT, scale=None, normk=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_abc_bwd_kernel_dh[grid](
                q, z, do, dh,
                q.stride(1), q.stride(2), q.stride(3),
                do.stride(1), do.stride(2), do.stride(3),
                dh.stride(1), dh.stride(2), dh.stride(3),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                SCALE=(scale is not None),
                NORMK=normk,
                num_warps=2,
                num_stages=num_stages
            )
            return dh

        def bwd_post(s, z, ss, B, H, T, S, BT, BC, BS, NT, NC, NS):
            doo = torch.empty_like(s)
            grid = (NS, B * H)
            chunk_abc_bwd_kernel_rcum_inter[grid](
                s, z, ss, doo,
                s.stride(1), s.stride(2), s.stride(3),
                T=T, S=S, BT=BT, BS=BS, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages
            )
            grid = (NS, NT * NC, B * H)
            chunk_abc_bwd_kernel_rcum_intra[grid](
                s, z, ss, doo,
                s.stride(1), s.stride(2), s.stride(3),
                T=T, S=S, BT=BT, BC=BC, BS=BS, NC=NC,
                num_warps=2,
                num_stages=num_stages
            )
            return doo

        dhv = bwd_inner(
            p, z, dov,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
            scale=None,
            normk=True
        )
        dp1 = torch.empty_like(p)
        dsv1 = torch.empty_like(s, dtype=torch.float)
        dv = v.new_empty(NM, *v.shape)
        dAv = q.new_zeros(B, H, T, BT)
        grid = (NM, NT, B * H)
        chunk_abc_bwd_kernel_V[grid](
            s, v, hv, z, Av, dov, dhv, dp1, dsv1, dv, dAv,
            p.stride(1), p.stride(2), p.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            hv.stride(1), hv.stride(2), hv.stride(3),
            T=T, K=M, V=V, BT=BT, BK=BM, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dv = dv.sum(0)
        dp0 = torch.empty_like(p)
        dsv0 = s.new_zeros(NC, *s.shape, dtype=torch.float)
        grid = (NM, NT * NC, B * H)
        chunk_abc_bwd_kernel_intra_V[grid](
            p, s, z, dAv, dp0, dsv0,
            p.stride(1), p.stride(2), p.stride(3),
            T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages
        )
        dp = dp1.add_(dp0)
        ds = dsv1.add_(dsv0.sum(0))

        # grad of softmax
        dok = torch.empty_like(p)
        grid = (NM, NT, B * H)
        softmax_bwd_kernel[grid](
            p, ov, dp, dov, dok,
            ov.stride(1), ov.stride(2), ov.stride(3),
            p.stride(1), p.stride(2), p.stride(3),
            T=T, V=V, S=M, BT=BT, BV=BV, BS=BM,
            num_warps=2,
            num_stages=num_stages
        )
        dhk = bwd_inner(
            q, z, dok,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
            scale=scale,
            normk=False
        )
        dAk = q.new_zeros(NM, B, H, T, BT)
        grid = (NM, NT * NC * NC, B * H)
        chunk_abc_bwd_kernel_intra_K[grid](
            s, z, dok, dAk,
            s.stride(1), s.stride(2), s.stride(3),
            scale,
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages
        )
        dAk = dAk.sum(0)

        Ak = q.new_zeros(NK, B, H, T, BT)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dsk1 = s.new_empty(NK, *s.shape, dtype=torch.float)
        grid = (NK, NT, B * H)
        chunk_abc_bwd_kernel_K[grid](
            q, k, s, hk, z, Ak, dok, dhk, dq, dk, dsk1, dAk,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            hk.stride(1), hk.stride(2), hk.stride(3),
            scale,
            T=T, K=K, V=M, BT=BT, BK=BK, BV=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        Ak = Ak.sum(0)
        dsk1 = dsk1.sum(0)
        dsk0 = torch.empty_like(s, dtype=torch.float)
        grid = (NM, NT * NC, B * H)
        chunk_abc_bwd_kernel_intra_KV[grid](
            s, z, Ak, dok, dsk0,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
            num_warps=2,
            num_stages=num_stages
        )
        ds.add_(dsk1.add_(dsk0))

        ds -= bwd_post(s, z, ok * dok + p * dp, B, H, T, M, BT, BC, BM, NT, NC, NM)
        ds = ds.to(s.dtype)
        return dq, dk, dv, ds


chunk_abc = ChunkABCFunction.apply

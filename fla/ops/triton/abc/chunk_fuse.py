# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous


@triton.jit
def chunk_abc_fwd_kernel_s(
    q,
    k,
    s,
    rk,  # rescale term
    ck,  # scores normalized over a chunk
    pk,  # scores normalized over the sequence
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_s = tl.make_block_ptr(s + (i_k * n_bh + i_bh)*s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_pk = tl.make_block_ptr(pk + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    b_hk = tl.zeros([BK, BM], dtype=tl.float32)
    for _ in range(NT):
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM,]
        b_rk = tl.load(p_rk, boundary_check=(0,))
        # [BT, BM]
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1))

        # [BT, BM]
        b_inter = tl.dot(b_q, b_hk.to(b_q.dtype), allow_tf32=False) * b_rk[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False), 0).to(b_q.dtype), b_ck, allow_tf32=False)
        b_s = (b_inter + b_intra) * b_pk
        # [BK, BM]
        b_hk = b_hk * b_rk[None, :] + tl.dot(b_k, b_ck, allow_tf32=False)

        tl.store(p_s, b_s.to(p_s.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_s = tl.advance(p_s, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (BT, 0))
        p_pk = tl.advance(p_pk, (BT, 0))


@triton.jit
def chunk_abc_fwd_kernel_o(
    p,
    v,
    o,
    rv,  # rescale term
    cv,  # scores normalized over a chunk
    pv,  # scores normalized over the sequence
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    DM: tl.constexpr,
    DV: tl.constexpr,
    NT: tl.constexpr
):
    i_v, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_m * n_bh + i_bh)*s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, 0), (BM, BT), (0, 1))
    p_pv = tl.make_block_ptr(pv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, BV]
    b_hv = tl.zeros([BM, BV], dtype=tl.float32)
    for _ in range(NT):
        # [BT, BM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BM,]
        b_rv = tl.load(p_rv, boundary_check=(0,))
        # [BM, BT]
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        # [BT, BM]
        b_pv = tl.load(p_pv, boundary_check=(0, 1))

        b_p = b_p * b_pv
        # [BT, BV]
        b_inter = tl.dot((b_p * b_rv[None, :]).to(b_v.dtype), b_hv.to(b_v.dtype), allow_tf32=False)
        b_intra = tl.where(m_s, tl.dot(b_p.to(b_v.dtype), b_cv, allow_tf32=False), 0)
        b_intra = tl.dot(b_intra.to(b_v.dtype), b_v, allow_tf32=False)
        b_o = b_inter + b_intra
        # [BM, BV]
        b_hv = b_hv * b_rv[:, None] + tl.dot(b_cv, b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        p_p = tl.advance(p_p, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_rv = tl.advance(p_rv, (DM,))
        p_cv = tl.advance(p_cv, (0, BT))
        p_pv = tl.advance(p_pv, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dp(
    v,
    rv,  # rescale term
    cv,  # scores normalized over a chunk
    pv,  # scores normalized over the sequence
    do,
    dp,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    DV: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (i_v * BV, 0), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_pv = tl.make_block_ptr(pv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_dp = tl.make_block_ptr(dp + (i_v * n_bh + i_bh)*s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV, BM]
    b_hv = tl.zeros([BV, BM], dtype=tl.float32)
    for _ in range(NT):
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BM,]
        b_rv = tl.load(p_rv, boundary_check=(0,))
        # [BT, BM]
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_pv = tl.load(p_pv, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BM]
        b_inter = tl.dot(b_do, b_hv.to(b_do.dtype), allow_tf32=False) * b_rv[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False), 0).to(b_v.dtype), b_cv, allow_tf32=False)
        b_dp = (b_inter + b_intra) * b_pv
        # [BV, BM]
        b_hv = b_hv * b_rv[None, :] + tl.dot(b_v, b_cv, allow_tf32=False)

        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.advance(p_v, (0, BT))
        p_rv = tl.advance(p_rv, (DM,))
        p_cv = tl.advance(p_cv, (BT, 0))
        p_pv = tl.advance(p_pv, (BT, 0))
        p_do = tl.advance(p_do, (BT, 0))
        p_dp = tl.advance(p_dp, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dq(
    k,
    rk,  # rescale term
    ck,  # scores normalized over a chunk
    dq,
    ds,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, 0), (BM, BT), (0, 1))
    p_dq = tl.make_block_ptr(dq + (i_m * n_bh + i_bh)*s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, BK]
    b_hk = tl.zeros([BM, BK], dtype=tl.float32)
    for _ in range(NT):
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM,]
        b_rk = tl.load(p_rk, boundary_check=(0,))
        # [BM, BT]
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        # [BT, BM]
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, BK]
        b_inter = tl.dot((b_ds * b_rk[None, :]).to(b_k.dtype), b_hk.to(b_k.dtype), allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ds, b_ck, allow_tf32=False), 0).to(b_k.dtype), b_k, allow_tf32=False)
        b_dq = b_inter + b_intra
        # [BM, BK]
        b_hk = b_hk * b_rk[:, None] + tl.dot(b_ck, b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

        p_k = tl.advance(p_k, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (0, BT))
        p_dq = tl.advance(p_dq, (BT, 0))
        p_ds = tl.advance(p_ds, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dk(
    q,
    k,
    rk,  # rescale term
    ck,  # scores normalized over a chunk
    ds,
    dk,
    dsk,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), ((NT-1)*BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, (NT-1)*BT), (BK, BT), (0, 1))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, (NT-1)*BT), (BM, BT), (0, 1))
    p_dk = tl.make_block_ptr(dk + (i_m*n_bh+i_bh)*s_qk_h, (T, DK), (s_qk_t, s_qk_d), ((NT-1)*BT, i_k * BK), (BT, BK), (1, 0))
    p_dsk = tl.make_block_ptr(dsk + (i_k*n_bh+i_bh)*s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]

    # [BM, BK]
    b_dhk = tl.zeros([BM, BK], dtype=tl.float32)
    for i in range(NT):
        p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (((NT-i) % NT) * DM + i_m * BM,), (BM,), (0,))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM,]
        b_rk = tl.load(p_rk, boundary_check=(0,))
        # [BT, BM]
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, BK]
        b_inter = tl.dot((b_ck * b_rk[None, :]).to(b_q.dtype), b_dhk.to(b_q.dtype), allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ck, b_ds, allow_tf32=False), 0.).to(b_q.dtype), b_q, allow_tf32=False)
        b_dk = b_inter + b_intra

        # [BM, BT]
        b_inter = tl.dot(b_dhk.to(b_k.dtype), b_k, allow_tf32=False) * b_rk[:, None]
        b_intra = tl.dot(b_ds, tl.where(m_t, tl.dot(b_q, b_k, allow_tf32=False), 0.).to(b_q.dtype), allow_tf32=False)
        # [BT, BM]
        b_dsk = b_ck * tl.trans(b_inter + b_intra)

        # [BM, BK]
        b_dhk = b_dhk * b_rk[:, None] + tl.dot(b_ds, b_q, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dsk, b_dsk.to(p_dsk.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (-BT, 0))
        p_k = tl.advance(p_k, (0, -BT))
        p_ck = tl.advance(p_ck, (-BT, 0))
        p_ds = tl.advance(p_ds, (0, -BT))
        p_dk = tl.advance(p_dk, (-BT, 0))
        p_dsk = tl.advance(p_dsk, (-BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dv(
    do,
    v,
    rv,  # rescale term
    cv,  # scores normalized over a chunk
    p,
    dv,
    dsv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    DV: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_v, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), ((NT-1)*BT, i_v * BV), (BT, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (i_v * BV, (NT-1)*BT), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, (NT-1)*BT), (BM, BT), (0, 1))
    p_dv = tl.make_block_ptr(dv + (i_m*n_bh+i_bh)*s_qk_h, (T, DV), (s_qk_t, s_qk_d), ((NT-1)*BT, i_v * BV), (BT, BV), (1, 0))
    p_dsv = tl.make_block_ptr(dsv + (i_v*n_bh+i_bh)*s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]

    # [BM, BV]
    b_dhv = tl.zeros([BM, BV], dtype=tl.float32)
    for i in range(NT):
        p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (((NT-i) % NT) * DM + i_m * BM,), (BM,), (0,))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BM,]
        b_rv = tl.load(p_rv, boundary_check=(0,))
        # [BT, BM]
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        # [BM, BT]
        b_p = tl.load(p_p, boundary_check=(0, 1))

        # [BT, BV]
        b_inter = tl.dot((b_cv * b_rv[None, :]).to(b_do.dtype), b_dhv.to(b_do.dtype), allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_cv, b_p, allow_tf32=False), 0.).to(b_do.dtype), b_do, allow_tf32=False)
        b_dv = b_inter + b_intra

        b_inter = tl.dot(b_dhv.to(b_v.dtype), b_v, allow_tf32=False) * b_rv[:, None]
        b_intra = tl.dot(b_p, tl.where(m_t, tl.dot(b_do, b_v, allow_tf32=False), 0.).to(b_do.dtype), allow_tf32=False)
        # [BT, BM]
        b_dsv = b_cv * tl.trans(b_inter + b_intra)

        # [BM, BV]
        b_dhv = b_dhv * b_rv[:, None] + tl.dot(b_p, b_do, allow_tf32=False)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dsv, b_dsv.to(p_dsv.dtype.element_ty), boundary_check=(0, 1))

        p_do = tl.advance(p_do, (-BT, 0))
        p_v = tl.advance(p_v, (0, -BT))
        p_cv = tl.advance(p_cv, (-BT, 0))
        p_p = tl.advance(p_p, (0, -BT))
        p_dv = tl.advance(p_dv, (-BT, 0))
        p_dsv = tl.advance(p_dsv, (-BT, 0))


@triton.jit
def chunk_abc_fwd_kernel_cum(
    s,
    r,
    c,
    p,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))

    b_mp = tl.zeros([BM,], dtype=tl.float32)
    b_zp = tl.zeros([BM,], dtype=tl.float32)
    for i in range(NT):
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

        b_m = tl.max(b_s, 0)
        # workaround for some weird compiler bugs
        if i == 0:
            b_r = tl.exp(-b_m)
        else:
            b_m = tl.maximum(b_mp, b_m)
            b_r = tl.exp(b_mp - b_m)
        b_c = tl.exp(b_s - b_m[None, :])
        b_z = tl.cumsum(b_c, 0) + (b_zp * b_r)[None, :]
        b_p = tl.exp(-tl.log(b_z))
        b_mp = b_m
        b_zp = tl.max(b_z, 0)

        tl.store(p_r, b_r.to(p_r.dtype.element_ty), boundary_check=(0,))
        tl.store(p_c, b_c.to(p_c.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))

        p_s = tl.advance(p_s, (BT, 0))
        p_r = tl.advance(p_r, (DM,))
        p_c = tl.advance(p_c, (BT, 0))
        p_p = tl.advance(p_p, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_rcum(
    s,
    r,
    c,
    o,
    s_sk_h,
    s_sk_t,
    s_sk_m,
    T,
    BT: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr,
    NT: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT-1)*BT, i_m * BM), (BT, BM), (1, 0))

    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_t = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BM,], dtype=tl.float32)
    for i in range(NT):
        p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (((NT-i) % NT) * DM + i_m * BM,), (BM,), (0,))
        # [BT, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        # [BM,]
        b_r = tl.load(p_r, boundary_check=(0,))
        # [BT, BM]
        b_c = tl.load(p_c, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))

        b_z = b_z * b_r
        b_o -= b_c * (b_z[None, :] + tl.dot(m_t.to(b_s.dtype), b_s, allow_tf32=False))

        # [BM,]
        b_z += tl.sum(b_s, 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        p_s = tl.advance(p_s, (-BT, 0))
        p_c = tl.advance(p_c, (-BT, 0))
        p_o = tl.advance(p_o, (-BT, 0))


class FusedChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        batch_size, n_heads, seq_len, d_head_qk, d_head_v, n_slots = *q.shape, v.shape[-1], sk.shape[-1]
        scale = d_head_qk ** -0.5

        DT, DK, DV, DM = seq_len, d_head_qk, d_head_v, n_slots
        BT = 16
        if batch_size * n_heads > 100:
            BK, BV, BM = min(DK, 64), min(DV, 64), min(DM, 64)
            num_stages = 1
            num_warps = 2
        else:
            # SM is not fully utilized so we add more parallelism in the hidden state dimension.
            BK, BV, BM = min(DK, 32), min(DV, 32), min(DM, 32)
            num_stages = 1
            num_warps = 1
        NT, NK, NV, NM = triton.cdiv(DT, BT), triton.cdiv(DK, BK), triton.cdiv(DV, BV), triton.cdiv(DM, BM)

        rk, ck, pk = sk.new_empty(batch_size, n_heads, NT, DM), torch.empty_like(sk), torch.empty_like(sk)
        grid = (NM, batch_size * n_heads)
        chunk_abc_fwd_kernel_cum[grid](
            sk, rk, ck, pk,
            sk.stride(1), sk.stride(2), sk.stride(3),
            seq_len,
            BT=BT, BM=BM, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        rv, cv, pv = sv.new_empty(batch_size, n_heads, NT, DM), torch.empty_like(sv), torch.empty_like(sv)
        chunk_abc_fwd_kernel_cum[grid](
            sv, rv, cv, pv,
            sv.stride(1), sv.stride(2), sv.stride(3),
            seq_len,
            BT=BT, BM=BM, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )

        s = q.new_empty(NK, batch_size, n_heads, seq_len, n_slots)
        grid = (NM, NK, batch_size * n_heads)
        chunk_abc_fwd_kernel_s[grid](
            q, k, s, rk, ck, pk,
            q.stride(1), q.stride(2), q.stride(3),
            sk.stride(1), sk.stride(2), sk.stride(3),
            seq_len, scale,
            BT=BT, BK=BK, BM=BM, DK=DK, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        s = s.sum(0)
        p = s.softmax(-1, dtype=torch.float).to(q.dtype)
        o = q.new_empty(NM, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NM, batch_size * n_heads)
        chunk_abc_fwd_kernel_o[grid](
            p, v, o, rv, cv, pv,
            q.stride(1), q.stride(2), q.stride(3),
            sk.stride(1), sk.stride(2), sk.stride(3),
            seq_len,
            BT=BT, BM=BM, BV=BV, DM=DM, DV=DV, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        o = o.sum(0)
        ctx.save_for_backward(q, k, v, o, s, p, rk, ck, pk, rv, cv, pv)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.n_slots = n_slots
        ctx.dtype = q.dtype
        ctx.scale = scale
        ctx.BT = BT
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, p, rk, ck, pk, rv, cv, pv = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk, d_head_v, n_slots = *q.shape, v.shape[-1], s.shape[-1]
        scale = d_head_qk ** -0.5

        DT, DK, DV, DM = seq_len, d_head_qk, d_head_v, n_slots
        BT = ctx.BT
        if batch_size * n_heads > 100:
            BK, BV, BM = min(DK, 64), min(DV, 64), min(DM, 64)
            num_stages = 1
            num_warps = 2
        else:
            BK, BV, BM = min(DK, 32), min(DV, 32), min(DM, 32)
            num_stages = 1
            num_warps = 2
        NT, NK, NV, NM = triton.cdiv(DT, BT), triton.cdiv(DK, BK), triton.cdiv(DV, BV), triton.cdiv(DM, BM)
        dp = s.new_empty(NV, *s.shape)
        grid = (NM, NV, batch_size * n_heads)
        chunk_abc_bwd_kernel_dp[grid](
            v, rv, cv, pv, do, dp,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BV=BV, BM=BM, DV=DV, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dp = dp.sum(0)
        ds = p * (dp - (o * do).sum(-1, True)) * pk
        dss = ds * scale
        dq, dk, dv = q.new_empty(NM, *q.shape), k.new_empty(NM, *k.shape), v.new_empty(NM, *v.shape)
        dsk, dsv = s.new_empty(NK, *s.shape), s.new_empty(NV, *s.shape)
        grid = (NK, NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dq[grid](
            k, rk, ck, dq, dss,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BK=BK, BM=BM, DK=DK, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        chunk_abc_bwd_kernel_dk[grid](
            q, k, rk, ck, dss, dk, dsk,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BK=BK, BM=BM, DK=DK, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dk, dsk = dk.sum(0), dsk.sum(0)

        p = p * pv
        grid = (NV, NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dv[grid](
            do, v, rv, cv, p, dv, dsv,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BV=BV, BM=BM, DV=DV, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dv, dsv = dv.sum(0), dsv.sum(0)
        grid = (NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_rcum[grid](
            ds * s, rk, ck, dsk,
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BM=BM, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        chunk_abc_bwd_kernel_rcum[grid](
            p * dp, rv, cv, dsv,
            s.stride(1), s.stride(2), s.stride(3),
            seq_len,
            BT=BT, BM=BM, DM=DM, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq, dk, dv, dsk, dsv


fused_chunk_abc = FusedChunkABCFunction.apply

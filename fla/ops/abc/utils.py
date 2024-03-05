# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import triton
import triton.language as tl


@triton.jit
def logcumsumexp_fwd_kernel(
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
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

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
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s) + b_zp
        # [BS,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
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
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [BT]
        b_mc = tl.max(b_s, 1)
        b_mc = tl.maximum(b_m, b_mc)
        if i > 0:
            b_z = b_z * tl.exp(b_m - b_mc)
        b_z += tl.sum(tl.exp(b_s - b_mc[:, None]), 1)
        b_m = b_mc

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_s = tl.exp(b_s - b_m[:, None])
    b_p = tl.where(b_s != 0, b_s / b_z[:, None], 0.)
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
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
    b_pp = tl.zeros([BT,], dtype=tl.float32)
    for i in range(tl.cdiv(S, BS)):
        p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i * BS), (BT, BS), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
        b_dp = tl.load(p_dp, boundary_check=(0, 1)).to(tl.float32)
        # [BT,]
        b_pp += tl.sum(b_p * b_dp, 1)
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
    b_dp = tl.load(p_dp, boundary_check=(0, 1)).to(tl.float32)
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))

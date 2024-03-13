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
    NT: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_mp = tl.full([S,], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S,], dtype=tl.float32)
    for i_t in range(NT):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

        # [BT, S]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [S,]
        b_mc = tl.max(b_s, 0)
        # workaround for compiler bugs
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        # [BT, S]
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        # [S,]
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
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

    # [BT, S]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    # [BT]
    b_m = tl.max(b_s, 1)

    # [BT, BS]
    b_s = tl.exp(b_s - b_m[:, None])
    b_z = tl.sum(b_s, 1)
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
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    # [BT, BS]
    b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
    b_dp = tl.load(p_dp, boundary_check=(0, 1)).to(tl.float32)
    # [BT,]
    b_pp = tl.sum(b_p * b_dp, 1)
    # [BT, BS]
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))

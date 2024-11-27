# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.gla.chunk import chunk_gla_bwd_dA, chunk_gla_bwd_dv
from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    o,
    o_minus_s,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o_minus_s = tl.make_block_ptr(o_minus_s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o_minus_s, (b_o - b_s).to(p_o_minus_s.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_fwd_cumsum(g, BT):
    B, H, T, K = g.shape
    NT = triton.cdiv(T, BT)
    g, gi, ge = g, torch.empty_like(g, dtype=torch.float), torch.empty_like(g, dtype=torch.float)
    def grid(meta): return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g, gi, ge,
        g.stride(1), g.stride(2), g.stride(3),
        T=T,
        S=K,
        BT=BT
    )
    return gi, ge


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    gi,  # cumulative decay inclusive
    ge,  # cumulative decay exclusive
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_i <= i_j:
        return
    if i_t * BT + i_i * BC >= T:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        # q block exlusive
        p_gq = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        # k block inclusive
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        # the last position of the k block inclusive
        p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,),
                                 ((i_t * BT + i_j * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.load(p_gq, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_gq - b_gn[None, :]) * scale)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[:, None] - b_gk))
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"],
)
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    gi,
    ge,
    u,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_i * BC >= T:
        return

    i_j = i_i
    i_h = i_bh % H
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    i_k = 0
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_u = tl.make_block_ptr(u + i_h * s_k_t, (s_k_t,), (1,), (i_k * BK), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))

    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T*K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        p_qi = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,),
                                 ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qi = tl.load(p_qi, boundary_check=(0,))
        A_jj = tl.sum(b_qi * b_k * b_u * scale)
        b_A = tl.where(o_i != j, b_A, A_jj)
        tl.store(A + o_A + j, b_A, mask=m_A)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    gi,
    ge,
    u,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    if i_t * BT + i_i * BC >= T:
        return

    i_j = i_i
    i_h = i_bh % H
    o_i = tl.arange(0, BC)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_u = tl.make_block_ptr(u + i_h * s_k_t, (s_k_t,), (1,), (i_k * BK), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))

    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T*K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        p_qi = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,),
                                 ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qi = tl.load(p_qi, boundary_check=(0,))
        A_jj = tl.sum(b_qi * b_k * b_u * scale)
        b_A = tl.where(o_i != j, b_A, A_jj)
        tl.store(A + o_A + j, b_A, mask=m_A)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC"],
)
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    T: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return

    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k*n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BV", "BT"],
)
@triton.jit
def chunk_rwkv6_fwd_kernel_inter(
    q,
    v,
    g,
    h,
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

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_ge = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_ge, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_A = tl.where(m_s, b_A, 0.)
    b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "NC", "BT"],
)
@triton.jit
def chunk_rwkv6_bwd_kernel_intra(
    q,
    k,
    gi,
    ge,
    dA,
    dq,
    dk,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_q = i_t * BT + i_i * BC
    m_k = o_k < K

    p_ge = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)

    if i_i > 0:
        b_gn = tl.load(gi + i_bh * T * K + (o_q - 1) * K + o_k, mask=(m_k & (i_i > 0) & (o_q <= T)), other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                    (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                     (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk))
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_ge - b_gn[None, :])

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        p_gkj = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, boundary_check=(0,)).to(tl.float32)
        b_gkj = tl.load(p_gkj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] > j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 for b_dA to have a good precision.
        tmp = tl.exp(b_ge - b_gkj[None, :])
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tmp, 0.)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_gk = tl.load(p_gk, boundary_check=(0, 1))

    max_block_idx = min(NC, tl.cdiv(T-i_t*BT, BC))
    if i_i < max_block_idx - 1:
        p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T*K,), (s_k_d,),
                                 ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                    (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_ge = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                     (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_ge = tl.load(p_ge, boundary_check=(0, 1))
            b_qg = b_q * tl.exp(b_ge - b_gn[None, :])
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK] fp32
            b_dk += tl.dot(tl.trans(b_dA), b_qg, allow_tf32=False)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gqj = tl.make_block_ptr(ge + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = tl.load(p_qj, boundary_check=(0,)).to(tl.float32)
        b_gqj = tl.load(p_gqj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] < j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        # triton.Config({}, num_warps=1),
        # triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BV", "BT"],
)
@triton.jit
def chunk_rwkv6_bwd_kernel_inter(
    q,
    k,
    v,
    h,
    gi,
    ge,
    u,
    do,
    dh,
    dA,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    du,
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
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)

    last_idx = min(T, i_t * BT + BT) - 1
    p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), (last_idx * K + i_k * BK,), (BK,), (0,))
    b_gn = tl.load(p_gn, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * V * K, (V, K),
                                 (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BK]
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
    p_gk = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gi = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * tl.exp(b_gn[None, :] - b_gi)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)

    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :] - b_q * b_dq

    o_i = tl.arange(0, BT)
    p_dA_dig = dA + i_bh * T * BT + (i_t * BT + o_i) * BT + o_i
    b_dA_dig = tl.load(p_dA_dig, mask=(i_t * BT + o_i) < T, other=0)
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    # scale is already applied to b_dA_diag
    b_dq += (b_dA_dig[:, None] * b_u[None, :] * b_k)
    b_dk += (b_dA_dig[:, None] * b_u[None, :] * b_q)
    b_du = tl.sum(b_dA_dig[:, None] * b_q * b_k, axis=0)
    p_du = tl.make_block_ptr(du + (i_h + i_t * n_bh) * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    tl.store(p_du, b_du, boundary_check=(0,))

    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) +  b_dgk[None, :]
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # work around triton compiler bugs.
    p_dq = tl.make_block_ptr(dq2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_fwd_intra_A_gated(q, k, gi, ge, u, scale, BT):
    BC = 16
    B, H, T, K = q.shape
    A = q.new_empty(B, H, T, BT, dtype=torch.float32)
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)
    grid = (triton.cdiv(T, BT), NC * NC, B * H)
    BK = min(64, triton.next_power_of_2(K))
    chunk_rwkv6_fwd_A_kernel_intra_sub_inter[grid](
        q, k, gi, ge, A,
        k.stride(1), k.stride(2), k.stride(3),
        scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
    )
    grid = (NT, NC, B * H)
    # TODO: can we merge the two kernels?
    # load the entire [BC, K] blocks into SRAM at once
    if K <= 256:
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra[grid](
            q, k, gi, ge, u, A,
            k.stride(1), k.stride(2), k.stride(3),
            scale,
            H=H, T=T, K=K, BT=BT, BC=BC, BK=triton.next_power_of_2(K), NC=NC
        )
    # split then merge
    else:
        BK = 128
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, H, T, BC, dtype=torch.float32)
        grid = (NK, NT * NC, B * H)
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split[grid](
            q, k, gi, ge, u, A_intra,
            k.stride(1), k.stride(2), k.stride(3),
            scale,
            H=H, T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
        )
        grid = (NT, NC, B * H)
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge[grid](
            A_intra, A,
            T=T, BT=BT, BC=BC, NK=NK
        )
    return A


def chunk_rwkv6_fwd_o_gated_gk(q, v, g_cumsum, A, h, BT, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    BV = min(32, triton.next_power_of_2(V))
    BK = min(32, triton.next_power_of_2(K))
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H)
    o = torch.empty_like(v)
    chunk_rwkv6_fwd_kernel_inter[grid](
        q, v, g_cumsum, h, o, A,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o


def chunk_rwkv6_bwd_dqk_intra(q, k, g_cumsum_inclusive, g_cumsum_exclusive, dA, BT, scale):
    B, H, T, K = q.shape
    BC = 16
    BK = min(64, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    grid = (NK, NT * NC, B * H)
    chunk_rwkv6_bwd_kernel_intra[grid](
        q, k, g_cumsum_inclusive, g_cumsum_exclusive, dA, dq, dk,
        k.stride(1), k.stride(2), k.stride(3), scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
    )
    return dq, dk


def chunk_rwkv6_bwd_dqkgu(q, k, v, h, g_cumsum_inclusive, g_cumsum_exclusive, u, do, dh, dA, dq, dk, BT, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    dg = torch.empty_like(g_cumsum_inclusive)
    BK = 64
    BV = 64
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    grid = (NK, NT, B * H)
    # work around triton compiler bugs.
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    du = torch.empty(NT, B, H, K, dtype=torch.float32, device=u.device)
    chunk_rwkv6_bwd_kernel_inter[grid](
        q, k, v, h, g_cumsum_inclusive, g_cumsum_exclusive, u, do, dh, dA, dq, dk, dq2, dk2, dg, du,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    du = du.sum([0, 1])
    return dq2, dk2, dg, du


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV"],
)
@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None
})
@triton.jit
def chunk_rwkv6_bwd_kernel_dh(
    q,
    gi,
    ge,
    do,
    dh,
    dht,
    dh0,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NG: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BV]
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_gk = tl.make_block_ptr(ge + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = (b_q * tl.exp(b_gk) * scale).to(b_q.dtype)
        p_gk_last = gi + i_bg * s_k_h + last_idx * K + i_k * BK + tl.arange(0, BK)
        p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
        b_dh *= tl.exp(b_gk_last)[:, None]
        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_bwd_dh(q, k, v, g_cumsum_inclusive, g_cumsum_exclusive, do, h0, dht, BT, scale, states_in_fp32=False):
    HQ = q.shape[1]
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    NG = HQ // H

    dh = k.new_empty(B, HQ, NT * K, V, dtype=k.dtype if not states_in_fp32 else torch.float32)
    if h0 is not None:
        dh0 = torch.empty_like(h0, dtype=torch.float32) if h0.requires_grad else None
    else:
        dh0 = None
    chunk_rwkv6_bwd_kernel_dh[(NK, NV, B * HQ)](
        q, g_cumsum_inclusive, g_cumsum_exclusive, do, dh, dht, dh0,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT, NG=NG
    )
    return dh, dh0


class ChunkRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, u, scale, initial_state, output_final_state):
        BT = 64
        g_cumsum_inclusive, g_cumsum_exclusive = chunk_rwkv6_fwd_cumsum(g, BT=BT)  # gi, ge for short
        h, ht = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum_inclusive,
            gv=None,
            h0=initial_state,
            output_final_state=output_final_state,
            states_in_fp32=False,
            chunk_size=BT
        )
        A = chunk_rwkv6_fwd_intra_A_gated(q, k, g_cumsum_inclusive, g_cumsum_exclusive, u, scale, BT)
        o = chunk_rwkv6_fwd_o_gated_gk(q, v, g_cumsum_exclusive, A, h, BT, scale)
        ctx.save_for_backward(q, k, v, g, initial_state, A, u)
        ctx.BT = BT
        ctx.scale = scale
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, A, u = ctx.saved_tensors
        BT, scale = ctx.BT, ctx.scale
        g_cumsum_inclusive, g_cumsum_exclusive = chunk_rwkv6_fwd_cumsum(g, BT=BT)  # gi, ge for short
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum_inclusive,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            states_in_fp32=True,
            chunk_size=BT
        )
        dh, dh0 = chunk_rwkv6_bwd_dh(
            q=q,
            k=k,
            v=v,
            g_cumsum_inclusive=g_cumsum_inclusive,
            g_cumsum_exclusive=g_cumsum_exclusive,
            do=do,
            h0=initial_state,
            dht=dht,
            BT=BT,
            scale=scale,
            states_in_fp32=True
        )
        # dq dk in fp32
        dA = chunk_gla_bwd_dA(v=v, do=do, scale=scale, chunk_size=BT)
        dv = chunk_gla_bwd_dv(k=k, g_cumsum=g_cumsum_inclusive, A=A, do=do, dh=dh, scale=scale, chunk_size=BT)
        dq, dk = chunk_rwkv6_bwd_dqk_intra(
            q=q,
            k=k,
            g_cumsum_inclusive=g_cumsum_inclusive,
            g_cumsum_exclusive=g_cumsum_exclusive,
            dA=dA,
            BT=BT,
            scale=scale
        )
        dq, dk, dg, du = chunk_rwkv6_bwd_dqkgu(
            q=q,
            k=k,
            v=v,
            h=h,
            g_cumsum_inclusive=g_cumsum_inclusive,
            g_cumsum_exclusive=g_cumsum_exclusive,
            u=u,
            do=do,
            dh=dh,
            dA=dA,
            dq=dq,
            dk=dk,
            BT=BT,
            scale=scale
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), du.to(u), None, dh0, None


def chunk_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        u (torch.Tensor):
            bonus representations of shape `[H]`.
        scale (Optional[int]):
            Scale factor for the rwkv6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` and `head_first=True` else `[B, H, M, V]`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v, g = map(lambda x: x.transpose(1, 2) if x is not None else None, (q, k, v, g))
    o, final_state = ChunkRWKV6Function.apply(q, k, v, g, u, scale, initial_state, output_final_state)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state

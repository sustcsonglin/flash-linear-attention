# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh_fn, chunk_fwd_h_fn
from fla.ops.utils import chunk_local_cumsum
from fla.utils import contiguous


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
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
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
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
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
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))

    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K


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
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return


    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K


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
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
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
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
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
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
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
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    s_k_h,
    s_k_t,
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
    m_k = o_k < K

    # [BC, BK]
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk))
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_g - b_gn[None, :])
    
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    p_kj = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gkj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.)
        p_kj += K
        p_gkj += K
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    max_block_idx = min(NC, tl.cdiv(T-i_t*BT,BC))
    if i_i < max_block_idx - 1:    
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC + BC - 1) * K + o_k, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, max_block_idx):
            p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_qg = (b_q * tl.exp(b_g - b_gn[None, :]))
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    p_qj = tl.max_contiguous(tl.multiple_of(q + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gqj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)

    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)
        p_qj += K
        p_gqj += K
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BV", "BT"],
)
@triton.jit
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    s_v_h,
    s_v_t,
    scale,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


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
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
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
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.)
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)

    last_idx = min(i_t * BT + BT, T) - 1

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + last_idx * K + o_k, BK), BK)

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT, BV]
        # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


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
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
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
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    last_idx = min(T, i_t * BT + BT) - 1
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + last_idx * K + o_k, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
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
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(b_gn[None, :] - b_gk)
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # work around triton compiler bugs.
    p_dq = tl.make_block_ptr(dq2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT):
    B, H, T, K = q.shape
    BC = 16
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)

    BK = min(64, triton.next_power_of_2(K))
    A = q.new_empty(B, H, T, BT, dtype=torch.float32)
    grid = (NT, NC * NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        k.stride(1), k.stride(2),
        scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
    )
    grid = (NT, NC, B * H)
    # load the entire [BC, K] blocks into SRAM at once
    if K <= 256:
        BK = triton.next_power_of_2(K)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
            q, k, g, A,
            k.stride(1), k.stride(2),
            scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK
        )
    # split then merge
    else:
        BK = 128
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, H, BT, BC, dtype=torch.float32)
        grid = (NK, NT * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
            q, k, g, A_intra,
            k.stride(1), k.stride(2),
            scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
        )
        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
            A_intra, A,
            T=T, BT=BT, BC=BC, NK=NK
        )
    return A


def chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    BK = min(32, triton.next_power_of_2(K))
    BV = min(32, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)

    grid = (NV, NT, B * H)
    o = torch.empty_like(v)
    chunk_gla_fwd_kernel_o[grid](
        q, v, g_cumsum, h, o, A,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o


def chunk_gla_bwd_dA_fn(v, do, BT, scale):
    B, H, T, V = v.shape
    NT = triton.cdiv(T, BT)
    dA = v.new_empty(B, H, T, BT, dtype=torch.float32)
    grid = (NT, B * H)
    chunk_gla_bwd_kernel_dA[grid](
        v, do, dA,
        v.stride(1), v.stride(2),
        scale,
        T=T, V=V, BT=BT, BV=min(64, triton.next_power_of_2(V))
    )
    return dA


def chunk_gla_bwd_dv_fn(k, g_cumsum, A, do, dh, BT, scale):
    B, H, T, K, V = *k.shape, do.shape[-1]
    BV = 32
    NT = triton.cdiv(T, BT)
    grid = (triton.cdiv(V, BV), NT, B * H)
    dv = torch.empty_like(do)
    chunk_gla_bwd_kernel_dv[grid](
        k, g_cumsum, A, do, dh, dv,
        k.stride(1), k.stride(2),
        do.stride(1), do.stride(2),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BV=BV, BK=64
    )
    return dv


def chunk_gla_bwd_dqk_intra_fn(q, k, g_cumsum, dA, BT):
    B, H, T, K = q.shape
    BC = 16
    BK = min(64, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    grid = (NK, NT * NC, B * H)
    chunk_gla_bwd_kernel_intra[grid](
        q, k, g_cumsum, dA, dq, dk,
        k.stride(1), k.stride(2),
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
    )
    return dq, dk


def chunk_gla_bwd_dqkg_fn(q, k, v, h, g_cumsum, do, dh, dq, dk, BT, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    BK = 64
    BV = 64
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)

    dg = torch.empty_like(g_cumsum)
    grid = (NK, NT, B * H)
    # work around triton compiler bugs.
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    chunk_gla_bwd_kernel_inter[grid](
        q, k, v, h, g_cumsum, do, dh, dq, dk, dq2, dk2, dg,
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return dq2, dk2, dg


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        BT = 64
        g_cumsum = chunk_local_cumsum(g, BT=BT)
        h, ht = chunk_fwd_h_fn(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            BT=BT,
            h0=initial_state,
            output_final_state=output_final_state,
            states_in_fp32=False
        )
        # the intra A is kept in fp32
        # the computation has very marginal effect on the entire throughput
        A = chunk_fwd_intra_gated_gk_fn(q, k, g_cumsum, scale, BT)
        o = chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale)
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float32:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.BT = BT
        ctx.scale = scale
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        if g_cumsum is None:
            g_cumsum = chunk_local_cumsum(g, BT=ctx.BT)
        BT, scale = ctx.BT, ctx.scale
        h, _ = chunk_fwd_h_fn(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            BT=BT,
            h0=initial_state,
            output_final_state=False,
            states_in_fp32=True
        )
        dh, dh0 = chunk_bwd_dh_fn(
            q=q,
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            do=do,
            h0=initial_state,
            dht=dht,
            BT=BT,
            scale=scale,
            states_in_fp32=True
        )
        dv = chunk_gla_bwd_dv_fn(k=k, g_cumsum=g_cumsum, A=A, do=do, dh=dh, BT=BT, scale=scale)
        # dq dk in fp32
        dA = chunk_gla_bwd_dA_fn(v=v, do=do, BT=BT, scale=scale)
        dq, dk = chunk_gla_bwd_dqk_intra_fn(q=q, k=k, g_cumsum=g_cumsum, dA=dA, BT=BT)
        dq, dk, dg = chunk_gla_bwd_dqkg_fn(
            q=q,
            k=k,
            v=v,
            h=h,
            g_cumsum=g_cumsum,
            do=do,
            dh=dh,
            dq=dq,
            dk=dk,
            BT=BT,
            scale=scale
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        g (torch.Tensor):
            Forget gates of shape `(B, H, T, K)` applied to keys.
        scale (Optional[int]):
            Scale factor for the GLA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    return o, final_state

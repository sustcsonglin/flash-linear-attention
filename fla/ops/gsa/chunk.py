# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.gla.chunk import chunk_gla_bwd, chunk_gla_fwd
from fla.ops.utils import chunk_local_cumsum, softmax_bwd, softmax_fwd
from fla.utils import contiguous


@triton.jit
def chunk_gsa_fwd_k_kernel_intra(
    v,
    g,
    o,
    A,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC > T:
        return

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + min(i_t * BT + i_i * BC, T) * V + o_v, BV), BV)
    else:
        p_g = tl.make_block_ptr(g + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_b * T*H*V + min(i_t * BT + i_i * BC, T) * H*V + i_h * V + o_v, BV), BV)
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        if HEAD_FIRST:
            p_A = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            p_v = tl.make_block_ptr(v + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
            p_gv = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        else:
            p_A = tl.make_block_ptr(A+i_b*T*HQ*BT+i_hq*BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            p_v = tl.make_block_ptr(v+i_b*T*H*V+i_h*V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
            p_gv = tl.make_block_ptr(g+i_b*T*H*V+i_h*V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg)
    # [BC, BV]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_A = i_bh * T*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        if HEAD_FIRST:
            p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * T*V + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
            p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        else:
            p_v = tl.max_contiguous(tl.multiple_of(v + i_b * T*H*V + i_h * V + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
            p_gv = tl.max_contiguous(tl.multiple_of(g + i_b * T*H*V + i_h * V + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        # avoid 0 * inf = inf
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    if HEAD_FIRST:
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    else:
        p_o = tl.make_block_ptr(o + i_b*T*HQ*V + i_hq*V, (T, V), (HQ*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_k_kernel_inter(
    q,
    k,
    h,
    g,
    o,
    A,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    NT: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bg * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        else:
            p_q = tl.make_block_ptr(q + i_b * T*HQ*K + i_hq * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bg * NT*K*V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k)
    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_g = tl.make_block_ptr(g + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * T*HQ*V + i_hq * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_b*T*HQ*BT + i_hq*BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o = b_o * tl.exp(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_k_kernel_dA(
    v,
    g,
    do,
    dA,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC > T:
        return

    if HEAD_FIRST:
        p_dA = tl.make_block_ptr(dA+(i_v*n_bh+i_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    else:
        p_dA = tl.make_block_ptr(dA+((i_v*B+i_b)*T*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j*BC), (BC, BC), (1, 0))

    # [BC, BC]
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bg * T*V, (V, T), (1, V), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
            p_gv = tl.make_block_ptr(g + i_bg * T*V, (V, T), (1, V), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
            p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v+(i_b*T*H+i_h)*V, (V, T), (1, H*V), (i_v * BV, i_t*BT + i_j*BC), (BV, BC), (0, 1))
            p_gv = tl.make_block_ptr(g+(i_b*T*H+i_h)*V, (V, T), (1, H*V), (i_v * BV, i_t*BT + i_j*BC), (BV, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + (i_b*T*H+i_h)*V + (i_t*BT + i_i*BC) * H*V + o_v, BV), BV)
            p_g = tl.make_block_ptr(g+(i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do+(i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :]) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        if HEAD_FIRST:
            p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
            p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * T*V + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
            p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
        else:
            p_g = tl.make_block_ptr(g + (i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
            p_v = tl.max_contiguous(tl.multiple_of(v + (i_b*T*H+i_h)*V + (i_t*BT + i_j*BC) * H*V + o_v, BV), BV)
            p_gv = tl.max_contiguous(tl.multiple_of(g + (i_b*T*H+i_h)*V + (i_t*BT + i_j*BC) * H*V + o_v, BV), BV)
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            # [BV,]
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            # [BC,]
            b_dAj = tl.sum(b_do * b_v[None, :] * tl.exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += V
            p_gv += V
        b_dA = tl.where(m_dA, b_dA, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_k_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    dgv,
    dA,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    NT: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh) * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + (i_b * T*HQ+i_hq)*K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (i_b * T*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_A = tl.make_block_ptr(A + ((i_k*B+i_b)*T*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bg * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (o_t - 1) * V + o_v, BV), BV)
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dg = tl.make_block_ptr(dg + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dgv = tl.make_block_ptr(dgv + (i_k*n_bh+i_bh) * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_g = tl.make_block_ptr(g + (i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + (i_b*T*H+i_h)*V + (o_t - 1) * H*V + o_v, BV), BV)
            p_do = tl.make_block_ptr(do + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + ((i_k*B+i_b)*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dg = tl.make_block_ptr(dg + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dgv = tl.make_block_ptr(dgv+((i_k*B+i_b)*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * NT*K*V + i_t * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        m_v = o_v < V

        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_gv = tl.exp(b_gn[None, :] - b_g)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g) * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV]
        b_dg = tl.sum(tl.trans(b_h) * b_dh, 0) * tl.exp(b_gn)

        b_dh = b_dh.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_k.dtype))
        b_dk += tl.dot((b_v * b_gv).to(b_v.dtype), tl.trans(b_dh))
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh) * b_gv
        # [BV]
        b_dg += tl.sum(b_dv * b_v, 0)

        if i_k == 0:
            b_dgv = tl.load(p_dg, boundary_check=(0, 1)) + b_dg[None, :]
        else:
            b_dgv = tl.zeros([BT, BV], dtype=tl.float32) + b_dg[None, :]

        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if HEAD_FIRST:
        p_dA = tl.make_block_ptr(dA + i_bh * T*BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dA = tl.make_block_ptr(dA + (i_b * T*HQ + i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_b * T*HQ + i_hq)*K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_b * T*HQ + i_hq)*K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # [BT, BT]
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_k_kernel_intra_dvg(
    v,
    g,
    o,
    A,
    do,
    dv,
    dg,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC > T:
        return

    if HEAD_FIRST:
        p_gv = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (min(i_t * BT + i_i * BC + BC, T) - 1) * V + o_v, BV), BV)
    else:
        p_gv = tl.make_block_ptr(g + (i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + (i_b*T*H+i_h)*V + (min(i_t * BT + i_i * BC + BC, T)-1)*H*V + o_v, BV), BV)
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        if HEAD_FIRST:
            p_g = tl.make_block_ptr(g + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
            p_A = tl.make_block_ptr(A + i_bh * T*BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        else:
            p_g = tl.make_block_ptr(g+(i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
            p_A = tl.make_block_ptr(A+(i_b*T*HQ+i_hq)*BT, (BT, T), (1, HQ*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            p_do = tl.make_block_ptr(do+(i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_j*BC, i_v*BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :])).to(b_do.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do)
    b_dv *= tl.exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)

    if HEAD_FIRST:
        p_g = tl.max_contiguous(tl.multiple_of(g + i_bg * T*V + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
        p_A = tl.max_contiguous(tl.multiple_of(A + i_bh * T*BT + (i_t * BT + i_i * BC) * BT + o_c, BC), BC)
        p_do = tl.max_contiguous(tl.multiple_of(do + i_bh * T*V + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    else:
        p_g = tl.max_contiguous(tl.multiple_of(g + (i_b*T*H+i_h)*V + (i_t * BT + i_i * BC) * H*V + o_v, BV), BV)
        p_A = tl.max_contiguous(tl.multiple_of(A + (i_b*T*HQ+i_hq)*BT + (i_t*BT + i_i*BC) * HQ*BT + o_c, BC), BC)
        p_do = tl.max_contiguous(tl.multiple_of(do + (i_b*T*HQ+i_hq)*V + (i_t*BT + i_i*BC) * HQ*V + o_v, BV), BV)

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_A = tl.load(p_A)
        # [BV,]
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += V
        p_A += BT
        p_do += V
    if HEAD_FIRST:
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.make_block_ptr(v + i_bg * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    else:
        p_o = tl.make_block_ptr(o + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (i_b*T*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_dg = tl.make_block_ptr(dg + (i_b*T*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))

    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_gsa_fwd_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = 1.,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, A, h, ht, o = chunk_gla_fwd(
        q=q,
        k=k,
        v=v,
        g=None,
        g_cumsum=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return A, h, ht, o


def chunk_gsa_fwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BC = chunk_size, 16
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[1] if head_first else q.shape[2]
    NT = triton.cdiv(T, BT)
    NV = triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        h0=h0,
        output_final_state=output_final_state,
        states_in_fp32=False,
        head_first=head_first,
        chunk_size=BT
    )
    o = v.new_empty(B, *((HQ, T) if head_first else (T, HQ)), V)
    A = q.new_empty(B, *((HQ, T) if head_first else (T, HQ)), BT)
    grid = (NV, NT, B * HQ)
    chunk_gsa_fwd_k_kernel_inter[grid](
        q,
        k,
        h,
        g,
        o,
        A,
        scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NG=NG,
        NT=NT,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    grid = (NV, NT * NC, B * HQ)
    chunk_gsa_fwd_k_kernel_intra[grid](
        v,
        g,
        o,
        A,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return A, h, ht, o


def chunk_gsa_bwd_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = 1.,
    head_first: bool = True,
    chunk_size: int = 64
):
    dq, dk, dv, dg, dh0 = chunk_gla_bwd(
        q=q,
        k=k,
        v=v,
        g=None,
        g_cumsum=g,
        scale=scale,
        initial_state=h0,
        h=h,
        A=A,
        do=do,
        dht=dht,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dg, dh0


def chunk_gsa_bwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = 1.,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[1] if head_first else q.shape[2]
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=None,
            gv=g,
            h0=h0,
            output_final_state=False,
            states_in_fp32=False,
            head_first=head_first,
            chunk_size=chunk_size
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        do=do,
        h0=h0,
        dht=dht,
        scale=scale,
        states_in_fp32=True,
        head_first=head_first,
        chunk_size=BT
    )
    dA = q.new_empty(NV, B, *((HQ, T) if head_first else (T, HQ)), BT)
    grid = (NV, NT * NC * NC, B * HQ)
    chunk_gsa_bwd_k_kernel_dA[grid](
        v,
        g,
        do,
        dA,
        scale,
        B=B,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dA = dA.sum(0, dtype=dA.dtype)

    A = do.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), BT)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, *((HQ, T) if head_first else (T, HQ)), K)
    dv = v.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), V)
    dgv = g.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), V, dtype=torch.float)
    grid = (NK, NT, B * HQ)
    chunk_gsa_bwd_k_kernel_dqkvg[grid](
        q,
        k,
        v,
        h,
        g,
        A,
        do,
        dh,
        dq,
        dk,
        dv,
        dg,
        dgv,
        dA,
        scale,
        B=B,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NG=NG,
        NT=NT,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    dgv = dgv.sum(0, dtype=dgv.dtype)

    grid = (NV, NT * NC, B * HQ)
    chunk_gsa_bwd_k_kernel_intra_dvg[grid](
        v,
        g,
        o,
        A,
        do,
        dv,
        dg,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dg = dgv.add_(chunk_local_cumsum(dg, chunk_size=BT, reverse=True, head_first=head_first))

    return dq, dk, dv, dg, dh0


def chunk_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    Ak, hk, hkt, ok = chunk_gsa_fwd_k(
        q=q,
        k=k,
        v=s,
        g=g,
        h0=hk0,
        output_final_state=output_final_state,
        scale=scale,
        head_first=head_first,
        chunk_size=chunk_size
    )

    # p is kept in fp32 for safe softmax backward
    p = softmax_fwd(ok, dtype=torch.float)

    qv = p.to(q.dtype)
    Av, hv, hvt, ov = chunk_gsa_fwd_v(
        q=qv,
        k=s,
        v=v,
        g=g,
        scale=1.,
        initial_state=hv0,
        output_final_state=output_final_state,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return Ak, hk, hkt, ok, p, Av, hv, hvt, ov


def chunk_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    ok: torch.Tensor,
    p: torch.Tensor,
    A: Tuple[torch.Tensor, torch.Tensor],
    h: Tuple[torch.Tensor, torch.Tensor],
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    do: torch.Tensor,
    dht: Tuple[torch.Tensor, torch.Tensor],
    head_first: bool = True,
    chunk_size: int = 64
):
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state

    _, Av = A
    hk, hv = h
    dhkt, dhvt = dht

    qv = p.to(q.dtype)
    dqv, dsv, dv, dg, dhv0 = chunk_gsa_bwd_v(
        q=qv,
        k=s,
        v=v,
        g=g,
        h0=hv0,
        h=hv,
        A=Av,
        do=do,
        dht=dhvt,
        dg=None,
        scale=1.,
        head_first=head_first,
        chunk_size=chunk_size
    )

    # softmax gradient, equivalent to:
    # dok = qv * (dqv - (qv * dqv).sum(-1, True))
    dok = softmax_bwd(p, dqv, dtype=ok.dtype)

    dq, dk, dsk, dg, dhk0 = chunk_gsa_bwd_k(
        q=q,
        k=k,
        v=s,
        g=g,
        h0=hk0,
        h=hk,
        o=ok,
        do=dok,
        dht=dhkt,
        dg=dg,
        scale=scale,
        head_first=head_first,
        chunk_size=chunk_size
    )

    ds = dsv.add_(dsk)
    if q.shape[1] != k.shape[1]:
        dk, dv, ds, dg = map(lambda x: reduce(x, 'b (h g) ... -> b h ...', 'sum', h=k.shape[1]), (dk, dv, ds, dg))
    dg = dg.to(s.dtype)
    return dq, dk, dv, ds, dg, dhk0, dhv0


class ChunkGSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        hk0: Optional[torch.Tensor],
        hv0: Optional[torch.Tensor],
        output_final_state: bool,
        checkpoint_level: int,
        head_first: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        BT = 64
        g_org, g = g, chunk_local_cumsum(g, BT)
        Ak, hk, hkt, ok, p, Av, hv, hvt, ov = chunk_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            head_first=head_first,
            chunk_size=BT
        )

        if checkpoint_level >= 1:
            del g
            g = g_org
        if checkpoint_level > 1:
            del hk
            del hv
            hk, hv = None, None
        else:
            hk0, hv0 = None, None

        ctx.save_for_backward(q, k, v, s, g, ok, p, Av, hk0, hv0, hk, hv)
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.head_first = head_first
        ctx.BT = BT
        return ov, hkt, hvt

    @staticmethod
    @contiguous
    def backward(ctx, dov, dhkt=None, dhvt=None):
        q, k, v, s, g, ok, p, Av, hk0, hv0, hk, hv = ctx.saved_tensors
        scale = ctx.scale
        head_first = ctx.head_first
        BT = ctx.BT

        if ctx.checkpoint_level >= 1:
            g = chunk_local_cumsum(g, BT)
        dq, dk, dv, ds, dg, dhk0, dhv0 = chunk_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            ok=ok,
            p=p,
            A=(None, Av),
            h=(hk, hv),
            initial_state=(hk0, hv0),
            scale=scale,
            do=dov,
            dht=(dhkt, dhvt),
            head_first=head_first,
            chunk_size=BT
        )
        return dq, dk, dv, ds, dg, None, dhk0, dhv0, None, None, None


def chunk_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    checkpoint_level: Optional[int] = 2,
    head_first: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, HQ, T, K]` if `head_first=True` else `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
            If not provided, this function is equivalent to vanilla ABC.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`.
            Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `2`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`.
    """
    assert checkpoint_level in [0, 1, 2]
    if g is None:
        # TODO: this 3 steps took huge amount of time, ought to be optimized
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z).to(k.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    o, *final_state = ChunkGSAFunction.apply(
        q,
        k,
        v,
        s,
        g,
        scale,
        hk0,
        hv0,
        output_final_state,
        checkpoint_level,
        head_first
    )
    return o, final_state

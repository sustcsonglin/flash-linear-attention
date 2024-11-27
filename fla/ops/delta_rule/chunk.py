# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.delta_rule.wy_fast import (bwd_prepare_wy_repr,
                                        fwd_prepare_wy_repr, fwd_recompute_w_u)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_prepare_dv(
    q,
    k,
    do,
    dv,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + i_b * T*H*K + i_h * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)
        b_A += tl.dot(b_k, b_q, allow_tf32=False)

    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_do = tl.make_block_ptr(do + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_b * T*H*V+i_h*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    v_new,
    h,
    h0,
    ht,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new + i_bh * T*V, (T, V), (V, 1),
                                            (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_k = tl.make_block_ptr(k+i_b*T*H*K+i_h*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d+i_b*T*H*K+i_h*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v+i_b*T*H*V+i_h*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+i_b*T*H*V + i_h * V, (T, V), (H*V, 1),
                                            (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BC, BK]
            b_d = tl.load(p_d, boundary_check=(0, 1))
            # [BC, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype))
            # [BK, BV]
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        else:
            p_q = tl.make_block_ptr(q + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    b_s = tl.where(m_s, b_s, 0)
    if HEAD_FIRST:
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_v = tl.make_block_ptr(v + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if STORE_FINAL_STATE:
        p_dht = tl.make_block_ptr(dht + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2 + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_q = tl.make_block_ptr(q+i_b*T*H*K+i_h*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k+i_b*T*H*K+i_h*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d+i_b*T*H*K+i_h*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv+i_b*T*H*V+i_h*V, (T, V), (H*V, 1),
                                         (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do+i_b*T*H*V+i_h*V, (T, V), (H*V, 1),
                                         (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2+i_b*T*H*V+i_h*V, (T, V), (H*V, 1),
                                          (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            # [BT, V]
            b_do = tl.load(p_do, boundary_check=(0, 1))

            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)

            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d, b_dv.to(b_q.dtype), allow_tf32=False)
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_bwd_kernel_dqkw(
    q,
    k,
    v,
    w,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dw,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    o_i = tl.arange(0, BT)

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + i_b * T*H*K + i_h * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_b * T*H*V+i_h*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V, (V, NT * K), (1, V), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V, (V, NT * K), (1, V), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, b_dh, allow_tf32=False)

        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)

    # [BK, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0).to(b_q.dtype)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dq *= scale
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    if HEAD_FIRST:
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dq = tl.make_block_ptr(dq + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))


def chunk_delta_rule_fwd_prepare_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)

    dv = torch.empty_like(do)
    chunk_delta_rule_fwd_kernel_prepare_dv[(NT, B*H)](
        q, k, do, dv,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dv


def chunk_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32

    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BC = min(BT, BC)
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    v_new = torch.empty_like(u)
    chunk_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        d=w,
        v_new=v_new,
        h=h,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        HEAD_FIRST=head_first
    )
    return h, v_new


def chunk_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    dht: Optional[torch.Tensor],
    dh0: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, do.shape[-1]
    else:
        B, T, H, K, V = *q.shape, do.shape[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    # H100
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 64 if K <= 128 else 32

    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    dh = q.new_empty(B, H, NT * K, V)
    dv2 = torch.empty_like(dv)
    grid = (NK, NV, B * H)
    chunk_delta_rule_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        d=w,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        STORE_FINAL_STATE=dht is not None,
        USE_INITIAL_STATE=dh0 is not None,
        HEAD_FIRST=head_first
    )
    return dh, dh0, dv2


def chunk_delta_rule_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *q.shape, v_new.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v_new.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v_new)
    grid = (NV, NT, B * H)
    chunk_delta_rule_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v_new,
        h=h,
        o=o,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )

    return o


def chunk_delta_rule_bwd_dqkw(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    du: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, v_new.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v_new.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    grid = (NK, NT, B * H)
    chunk_delta_rule_bwd_kernel_dqkw[grid](
        q=q,
        k=k,
        v=v_new,
        w=w,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=du,
        dw=dw,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    return dq, dk, dw


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    checkpoint_level: int = 1,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # obtain WY representation. u is actually the new v.
    w, u, A = fwd_prepare_wy_repr(k, v, beta, head_first, BT)

    final_state = None
    if output_final_state:
        final_state = q.new_empty(B, H, K, V, dtype=torch.float)
    h, v_new = chunk_delta_rule_fwd_h(k, w, u, initial_state, final_state, head_first, BT)

    # obtain output
    o = chunk_delta_rule_fwd_o(q, k, v_new, h, scale, head_first, BT)
    if checkpoint_level == 1:
        h, v_new = None, None
    return o, A, h, v_new, final_state


def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    v_new: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    head_first: bool,
    chunk_size: int
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(k, v, beta, A, head_first, BT)
    if h is None:
        h, v_new = chunk_delta_rule_fwd_h(k, w, u, initial_state, None, head_first, BT)
    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None
    dv = chunk_delta_rule_fwd_prepare_dv(q, k, do, scale, head_first, BT)
    dh, dh0, dv = chunk_delta_rule_bwd_dhu(q, k, w, dht, dh0, do, dv, scale, head_first, BT)
    dq, dk, dw = chunk_delta_rule_bwd_dqkw(q, k, v_new, w, h, dv, do, dh, scale, head_first, BT)
    dk2, dv, db = bwd_prepare_wy_repr(k, v, beta, A, dw, dv, head_first, BT)
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        checkpoint_level=1,
        head_first=True
    ):
        BT = 64

        o, A, h, v_new, final_state = chunk_delta_rule_fwd(
            q,
            k,
            v,
            beta,
            scale,
            initial_state,
            output_final_state,
            checkpoint_level,
            head_first,
            chunk_size=BT
        )
        ctx.save_for_backward(q, k, v, beta, A, h, v_new, initial_state)
        ctx.BT = BT
        ctx.scale = scale
        ctx.head_first = head_first
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, beta, A, h, v_new, initial_state = ctx.saved_tensors
        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q,
            k,
            v,
            beta,
            A,
            h,
            v_new,
            ctx.scale,
            initial_state,
            do,
            dht,
            ctx.head_first,
            ctx.BT
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None


def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    checkpoint_level: int = 1,
    head_first: bool = True
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `1`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the forward hidden states during backward.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        checkpoint_level,
        head_first
    )
    return o, final_state

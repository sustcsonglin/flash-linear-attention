# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    h0,
    ht,
    offsets,
    c_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(c_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_v = tl.make_block_ptr(v + i_nh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
            d_i = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    offsets,
    indices,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    b_o = b_o * d_i[:, None]
    b_s = b_s * d_s
    if HEAD_FIRST:
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    do,
    dh,
    dh0,
    dht,
    offsets,
    c_offsets,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(c_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_do = tl.make_block_ptr(do + i_nh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dh = tl.make_block_ptr(dh + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        d_b = tl.math.exp2(min(BT, T - i_t * BT) * b_b)
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q,
    k,
    v,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    offsets,
    indices,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((min(BT, T - i_t * BT) - o_i - 1) * b_b)
    d_q = (d_q * scale).to(d_q.dtype)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (i_k * B*H + i_bh) * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + ((i_k*all+bos)*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

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
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    # [BT, BT]
    b_ds = (b_ds * d_s).to(b_q.dtype)
    # [BT, BK]
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    if HEAD_FIRST:
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dq = tl.make_block_ptr(dq + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_retention_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.Tensor] = None,
    c_offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, c_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        if c_offsets is None:
            c_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
        NT = c_offsets[-1]
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    if head_first:
        h = k.new_empty(B, H, NT, K, V, dtype=k.dtype)
    else:
        h = k.new_empty(B, NT, H, K, V, dtype=k.dtype)
    ht = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NK, NV, N * H)
    chunk_retention_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        h0=h0,
        ht=ht,
        offsets=offsets,
        c_offsets=c_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return h, ht


def chunk_retention_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, triton.next_power_of_2(T))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    grid = (NV, NT, B * H)
    chunk_retention_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        o=o,
        offsets=offsets,
        indices=indices,
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


def chunk_retention_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    offsets: Optional[torch.Tensor] = None,
    c_offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, c_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        if c_offsets is None:
            c_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
        NT = c_offsets[-1]
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    if head_first:
        dh = k.new_empty(B, H, NT, K, V, dtype=k.dtype)
    else:
        dh = k.new_empty(B, NT, H, K, V, dtype=k.dtype)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None

    grid = (NK, NV, N * H)
    chunk_retention_bwd_kernel_dh[grid](
        q=q,
        do=do,
        dh=dh,
        dh0=dh0,
        dht=dht,
        offsets=offsets,
        c_offsets=c_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dh, dh0


def chunk_retention_bwd_dqkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, triton.next_power_of_2(T))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape)
    grid = (NK, NT, B * H)
    chunk_retention_bwd_kernel_dqkv[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        offsets=offsets,
        indices=indices,
        scale=scale,
        B=B,
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
    dv = dv.sum(0)
    return dq, dk, dv


def chunk_retention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, ht = chunk_retention_fwd_h(
        k=k,
        v=v,
        h0=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=chunk_size
    )
    o = chunk_retention_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return o, ht


def chunk_retention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, _ = chunk_retention_fwd_h(
        k=k,
        v=v,
        h0=initial_state,
        output_final_state=False,
        offsets=offsets,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dh, dh0 = chunk_retention_bwd_dh(
        q=q,
        k=k,
        v=v,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        offsets=offsets,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dq, dk, dv = chunk_retention_bwd_dqkv(
        q=q,
        k=k,
        v=v,
        h=h,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dh0


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: torch.LongTensor,
        head_first: bool
    ):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, triton.next_power_of_2(T))

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        o, ht = chunk_retention_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), ht

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, initial_state = ctx.saved_tensors
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        dq, dk, dv, dh0 = chunk_retention_bwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, dh0, None, None, None


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        offsets (Optional[torch.LongTensor]):
            Offsets of shape `[N+1]` defining the bos/eos positions of `N` variable-length sequences in the batch.
            For example,
            if `offsets` is `[0, 1, 3, 6, 10, 15]`, there are `N=5` sequences with lengths 1, 2, 3, 4 and 5 respectively.
            If provided, the inputs are concatenated and the batch size `B` is expected to be 1.
            Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.retention import chunk_retention
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_retention(q, k, v,
                                    initial_state=h0,
                                    output_final_state=True,
                                    head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `offsets` is required
        >>> q, k, v = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v))
        # for a batch with 4 sequences, offsets with 5 start/end positions are expected
        >>> offsets = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_retention(q, k, v,
                                            initial_state=h0,
                                            output_final_state=True,
                                            offsets=offsets,
                                            head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if offsets is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `offsets`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(offsets) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(offsets) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkRetentionFunction.apply(
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        offsets,
        head_first
    )
    return o, final_state

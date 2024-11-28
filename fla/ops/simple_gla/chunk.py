# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
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
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_g = tl.make_block_ptr(g + i_b * T*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_v = tl.make_block_ptr(v + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dg,
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
    HEAD_FIRST: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_g_last = tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1)
    else:
        p_g = tl.make_block_ptr(g + i_b * T*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g_last = tl.load(g + i_b * T*H + (min(i_t * BT + BT, T) - 1) * H)
    b_g = tl.load(p_g, boundary_check=(0,))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V, (V, NT * K), (1, V), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V, (V, NT * K), (1, V), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dg_last += (tl.sum(b_h * b_dh))
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + (i_k*B*H + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_q = tl.make_block_ptr(q + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dq = tl.make_block_ptr(dq + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dk = tl.make_block_ptr(dk + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BK, BT), (0, 1))
        p_dg = tl.make_block_ptr(dg + (i_k*B + i_b) * T*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T-i_t*BT) - 1, b_dg, b_dg + b_dg_last)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dv(
    q,
    k,
    g,
    do,
    dv,
    dh,
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

    if HEAD_FIRST:
        b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))
        b_g_last = tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1)
    else:
        b_g = tl.load(g + i_b * T*H + (i_t * BT + tl.arange(0, BT)) * H + i_h)
        b_g_last = tl.load(g + i_b * T*H + (min(i_t * BT + BT, T) - 1) * H + i_h)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V, (NT * K, V), (V, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype)) * tl.exp(-b_g + b_g_last)[:, None]

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + i_b * T*H*K + i_h * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_b * T*H*K + i_h * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q, allow_tf32=False)

    b_A = b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0).to(do.dtype.element_ty)

    if HEAD_FIRST:
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_do = tl.make_block_ptr(do + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_b * T*H*V + i_h * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A, b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunk_simple_gla_fwd_o(
    h: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)

    o = torch.empty_like(v)
    grid = (NV, NT, B * H)
    chunk_simple_gla_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        scale,
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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT']
)
@triton.jit
def compute_final_dg(
    dg,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_o = tl.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_o = tl.load(p_o, boundary_check=(0,))
    b_o = b_o - tl.cumsum(b_o, axis=0) + tl.sum(b_o, axis=0)
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_simple_gla_bwd_dqkg(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device).fill_(-1e9)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q,
        k,
        v,
        h,
        g,
        do,
        dh,
        dq,
        dk,
        dg,
        scale,
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
    dg = dg.sum(0)
    dg2 = torch.empty(dg.shape, dtype=torch.float32, device=g.device)
    compute_final_dg[(NT, B*H)](dg, dg2, T=T, BT=BT)
    return dq, dk, dg2


def chunk_simple_gla_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    dv = torch.empty_like(do)
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    chunk_simple_gla_bwd_kernel_dv[(NV, NT, B*H)](
        q,
        k,
        g,
        do,
        dv,
        dh,
        scale,
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
    return dv


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(g, chunk_size, head_first=head_first)
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        head_first=head_first,
        chunk_size=chunk_size
    )
    o = chunk_simple_gla_fwd_o(
        h=h,
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return g, o, ht


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (SY 09/22) states_in_fp32 seems not affecting the error of dg but for safety, set to True
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        states_in_fp32=True,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        states_in_fp32=True,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dq, dk, dg = chunk_simple_gla_bwd_dqkg(
        do,
        q,
        k,
        v,
        g,
        h,
        dh,
        scale,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dv = chunk_simple_gla_bwd_dv(
        q,
        k,
        g,
        do,
        dh,
        scale,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state, head_first):
        BT = 64
        g, o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            head_first=head_first,
            chunk_size=BT
        )
        ctx.save_for_backward(q, k, v, g, initial_state)
        ctx.BT = BT
        ctx.scale = scale
        ctx.head_first = head_first
        return o.to(q.dtype), ht

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        BT, scale, head_first = ctx.BT, ctx.scale, ctx.head_first
        q, k, v, g, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            head_first=head_first,
            chunk_size=BT
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), None, dh0, None, None


def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
    scale: Optional[float] = None,
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
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for the attention scores.
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
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions"
    assert g.dim() == 3, "g must have 3 dimensions"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, head_first)
    return o, final_state

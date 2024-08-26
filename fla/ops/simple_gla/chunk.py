# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.ops.utils import chunk_local_cumsum, chunk_global_reversed_cumsum


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_h(
    k,
    v,
    h,
    g,
    h0,
    ht,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        if i_t < NT - 1:
            b_g_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)
        else:
            b_g_last = tl.load(g + i_bh * T + T - 1)
        b_h *= tl.exp(b_g_last)
        b_g = tl.load(p_g, boundary_check=(0,))
        b_h += tl.dot(b_k, (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_k.dtype), allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

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
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
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

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dh(
    q,
    g,
    do,
    dh,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_q = (b_q * scale * tl.exp(b_g)[None, :]).to(b_q.dtype)
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        if i_t == NT - 1:
            b_dh *= tl.exp(tl.load(g + i_bh * T + T - 1))
        else:
            b_dh *= tl.exp(tl.load(g + i_bh * T + i_t * BT + BT - 1))
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False)
    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    if i_t < NT - 1:
        b_g_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)
    else:
        b_g_last = tl.load(g + i_bh * T + T - 1)
    mask = tl.exp(b_g[None, :] - b_g[:, None])
    mask = tl.where(o_i[:, None] <= o_i[None, :], mask * scale, 0)
    b_s = b_s * mask
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
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
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * tl.exp(-b_g + b_g_last)[:, None]
        b_dv += tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dq = b_dq * tl.exp(b_g)[:, None]
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_ds = b_ds * tl.trans(mask)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    
    tl.debug_barrier()
    b_ds = None 
    b_s = None
    b_q = None
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) 
    b_dg = tl.sum(b_dq * b_q - b_dk * b_k.to(tl.float32), axis=1)
    p_dg = tl.make_block_ptr(dg + (i_k*n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    

def chunk_fwd_h_fn(k, v, g, BT, initial_state, output_final_state):
    B, H, T, K, V = *k.shape, v.shape[-1]
    final_state = None
    if output_final_state:
        final_state = k.new_empty(B, H, K, V, dtype=torch.float32)

    BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    h = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    chunk_simple_gla_fwd_kernel_h[grid](
        k, v, h, g, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state
    )
    return h, final_state


def chunk_fwd_o_fn(h, q, k, v, g, BT, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.empty_like(v)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H)
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o


def chunk_bwd_dh_fn(do, q, k, v, g, BT, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    dh = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    chunk_simple_gla_bwd_kernel_dh[grid](
        q, g, do, dh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
    )
    return dh


def chunk_bwd_dqkvg_fn(do, q, k, v, g, h, dh, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape)
    dg = torch.empty(NK, B, H, T, dtype=torch.float32, device=g.device)
    chunk_simple_gla_bwd_kernel_dqkvg[grid](
        q, k, v, h, g, do, dh, dq, dk, dv, dg,
        q.stride(1), q.stride(2), q.stride(3), 
        v.stride(1), v.stride(2), v.stride(3),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
    )
    dv = dv.sum(0)
    dg = dg.sum(0)
    dg = chunk_global_reversed_cumsum(dg)
    return dq, dk, dv, dg




class SimpleGLAFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state, checkpoint_level=1):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT = 64
        g = chunk_local_cumsum(g, BT)
        h, final_state = chunk_fwd_h_fn(k, v, g, BT, initial_state, output_final_state)
        o = chunk_fwd_o_fn(h, q, k, v, g, BT, scale)        
        if checkpoint_level == 1:
            h = None
        ctx.save_for_backward(q, k, v, h, g, initial_state)
        ctx.scale = scale
        ctx.BT = BT
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        BT, scale = ctx.BT, ctx.scale
        q, k, v, h, g, initial_state = ctx.saved_tensors
        if h is None:
            h, _ = chunk_fwd_h_fn(k, v, g, BT, initial_state, False)
        dh = chunk_bwd_dh_fn(do, q, k, v, g, BT, scale)
        dq, dk, dv, dg = chunk_bwd_dqkvg_fn(do, q, k, v, g, h, dh, scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), None, None, None, None



def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    checkpoint_level: int = 1
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
            Forget gates of shape `(B, H, T)` applied to keys.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `1` (recommended):
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the chunk-level hidden state `h` during backward pass.
    """
    assert checkpoint_level in [0, 1], "checkpoint_level must be 0, 1"
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions (b, h, l, d)"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    g = g.float()
    o, final_state = SimpleGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, checkpoint_level)
    return o, final_state
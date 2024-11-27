# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None
})
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    attn,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T), (T, 1), (i_t * BT, 0), (BT, BS), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_g = tl.load(p_g, boundary_check=(0,))

        b_gn = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gp = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.

        b_kg = (b_k * tl.exp(b_gn - b_g)).to(b_k.dtype)
        # [BT, BS]
        b_s = tl.dot(b_q, b_kg, allow_tf32=False)
        # do this check to avoid some layout bugs
        # [[BT, BV]
        if i_s > 0:
            b_o = b_o * tl.exp(b_gn - b_gp)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))

    tl.debug_barrier()

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT,]
    b_gq = tl.load(p_g, boundary_check=(0,))
    # rescale interchunk output
    b_o *= tl.exp(b_gq)[:, None]

    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k * B * H + i_bh) * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BS), (1, 0))

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap.
    # masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False) * tl.exp(b_gq[:, None] - b_gk[None, :]), 0)
        # [BT, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
        o_k += BS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_simple_gla_bwd_kernel_dq(
    i_bh,
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dq,
    dg,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_g = tl.load(p_g, boundary_check=(0,))

        b_gn = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gp = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.
        # [BT, BS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * tl.exp(b_gn - b_g)[None, :]
        # [BT, BK]
        if i_s > 0:
            b_dq *= tl.exp(b_gn - b_gp)
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k, allow_tf32=False)

    p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT,]
    b_gq = tl.load(p_gq, boundary_check=(0,))
    # [BT, BK]
    b_dq *= tl.exp(b_gq)[:, None] * scale

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap. masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False) * tl.exp((b_gq[:, None] - b_gk[None, :])), 0) * scale
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)

        o_k += BS
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (i_v * B * H + i_bh) * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_v * B * H + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg = tl.sum(b_dq * b_q, 1)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_simple_gla_bwd_kernel_dkv(
    i_bh,
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dk,
    dv,
    dg,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    # [BT,]
    b_gk = tl.load(p_gk, boundary_check=(0,))

    NTS = tl.cdiv(T, BS)
    # [BT, BK]
    b_kg = (b_k * tl.exp(tl.load(g + i_bh * T + min(i_t * BT + BT, T) - 1) - b_gk)[:, None]).to(b_k.dtype)

    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS,]
        b_gq = tl.load(p_gq, boundary_check=(0,))

        b_gp = tl.load(g + i_bh * T + min(i_s + BS, T) - 1)
        b_gn = tl.load(g + i_bh * T + i_s - 1) if i_s % BT > 0 else 0.
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_gq - b_gn)[:, None]).to(b_do.dtype)

        # overall decay rate for an entire block
        # [BS, BK]
        b_dk *= tl.exp(b_gp - b_gn)
        # [BS, BV]
        b_dv *= tl.exp(b_gp - b_gn)
        # [BT, BS]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_s = tl.dot(b_kg, tl.trans(b_q), allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q, allow_tf32=False)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do, allow_tf32=False)

    # [BT, BK]
    b_dk *= tl.exp(tl.load(g + i_bh * T + min(T, i_t * BT + BT) - 1) - b_gk)[:, None] * scale
    # [BT, BV]
    b_dv *= scale

    tl.debug_barrier()
    o_q = i_t * BT + tl.arange(0, BS)
    o_k = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_gq = tl.load(p_gq, boundary_check=(0,))
        # [BT, BS]
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.exp(-b_gk[:, None] + b_gq[None, :]), 0) * scale

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False) * d_s
        b_s = tl.dot(b_k, tl.trans(b_q), allow_tf32=False) * d_s
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q, allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        o_q += BS
    p_dk = tl.make_block_ptr(dk + (i_v * B * H + i_bh) * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_k * B * H + i_bh) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_v * B * H + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dg = tl.load(p_dg, boundary_check=(0,))
    b_dg -= tl.sum(b_dk * b_k, 1)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV'])
})
@triton.jit
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    do,
    dq,
    dk,
    dv,
    dg,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV

    parallel_simple_gla_bwd_kernel_dq(
        i_bh,
        i_t,
        i_k,
        i_v,
        q,
        k,
        v,
        g,
        do,
        dq,
        dg,
        s_k_h,
        s_k_t,
        s_v_h,
        s_v_t,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(
        i_bh,
        i_t,
        i_k,
        i_v,
        q,
        k,
        v,
        g,
        do,
        dk,
        dv,
        dg,
        s_k_h,
        s_k_t,
        s_v_h,
        s_v_t,
        scale,
        B,
        H,
        T,
        K,
        V,
        BT,
        BS,
        BK,
        BV
    )


def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    output_attentions: bool = False,
    chunk_size: int = 128
):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    num_stages = 3 if K <= 64 else 2
    num_warps = 4

    # local cumulative decay in log space
    g = chunk_local_cumsum(g, BT)

    grid = (NK * NV, triton.cdiv(T, BT), B * H)
    o = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None
    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        o=o,
        attn=attn,
        s_k_h=k.stride(1),
        s_k_t=k.stride(2),
        s_v_h=v.stride(1),
        s_v_t=v.stride(2),
        scale=scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_stages=num_stages,
        num_warps=num_warps
    )
    o = o.sum(0)
    if output_attentions:
        attn = attn.sum(0)
    return o, g, attn


def parallel_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int = 128
):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    BK = min(128, triton.next_power_of_2(k.shape[-1]))
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    num_stages = 3 if K <= 64 else 2
    num_warps = 4

    dq = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
    dv = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)
    dg = torch.empty(NV, B, H, T, dtype=torch.float, device=q.device)
    grid = (NK * NV, triton.cdiv(T, BT), B * H)
    parallel_simple_gla_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        s_k_h=k.stride(1),
        s_k_t=k.stride(2),
        s_v_h=v.stride(1),
        s_v_t=v.stride(2),
        scale=scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_stages=num_stages,
        num_warps=num_warps
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = dg.sum(0)
    dg = chunk_global_cumsum(dg, reverse=True)
    return dq, dk, dv, dg


class ParallelSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, output_attentions):
        BT = 128
        ctx.dtype = g.dtype
        o, g, attn = parallel_simple_gla_fwd(q, k, v, g, scale, output_attentions, BT)
        ctx.save_for_backward(q, k, v, g)
        ctx.scale = scale
        ctx.BT = BT
        return o.to(q.dtype), attn

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, da=None):
        q, k, v, g = ctx.saved_tensors
        dq, dk, dv, dg = parallel_simple_gla_bwd(q, k, v, g, do, ctx.scale, ctx.BT)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.dtype), None, None


def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    output_attentions: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]`
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` applied to keys.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q, k, v, g = map(lambda x: x.transpose(1, 2) if x is not None else None, (q, k, v, g))
    o, attn = ParallelSimpleGLAFunction.apply(q, k, v, g, scale, output_attentions)
    if not head_first:
        o = o.transpose(1, 2).contiguous()
    return o, attn

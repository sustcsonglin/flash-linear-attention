# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None
})
@triton.jit
def parallel_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    attn,
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
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    # [BS]
    o_k = tl.arange(0, BS)
    d_h = tl.math.exp2((BS - o_k) * b_b)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BS, BV), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k*B*H + i_bh) * T * T, (T, T), (T, 1), (i_t * BT, 0), (BT, BS), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for i in range(0, i_t * BT, BS):
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_h
        # do this check to avoid some layout bugs
        # [[BT, BV]
        if i > 0:
            b_o = b_o * tl.math.exp2(b_b * BS)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BS))
        p_v = tl.advance(p_v, (BS, 0))
        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))

    tl.debug_barrier()

    o_q = tl.arange(0, BT)
    d_q = tl.math.exp2(tl.arange(0, BT) * b_b)
    # rescale interchunk output
    b_o *= d_q[:, None]

    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BS, BV), (1, 0))
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + (i_k*B*H + i_bh) * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BS), (1, 0))

    # Q block and K block have overlap.
    # masks required
    for _ in range(i_t * BT, (i_t + 1) * BT, BS):
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BT, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
            p_a = tl.advance(p_a, (0, BS))
        p_k = tl.advance(p_k, (0, BS))
        p_v = tl.advance(p_v, (BS, 0))
        o_k += BS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_t*BT, i_v*BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_retention_bwd_kernel_dq(
    i_bh,
    i_t,
    i_k,
    i_v,
    i_h,
    k,
    v,
    do,
    dq,
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
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, 0), (BV, BS), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # overall decay rate for an entire block
    d_b = tl.math.exp2(b_b * BS)
    # cumulative decay from the end of the chunk
    d_h = tl.math.exp2((BS - tl.arange(0, BS)) * b_b)
    for i in range(0, i_t * BT, BS):
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        # [BT, BK]
        if i != 0:
            b_dq *= d_b
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (0, BS))
    b_dq *= tl.math.exp2(tl.arange(0, BT) * b_b)[:, None] * scale

    o_q = tl.arange(0, BT)
    o_k = tl.arange(0, BS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i_t * BT), (BV, BS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_t * BT, (i_t + 1) * BT, BS):
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (0, BS))
        o_k += BS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_t*BT, i_k*BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_retention_bwd_kernel_dkv(
    i_bh,
    i_t,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dk,
    dv,
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
    # no overlap. no need for mask.
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # overall decay rate for an entire block
    d_b = tl.math.exp2(b_b * BS)
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    NTS = tl.cdiv(T, BS)
    # [BT]
    d_h = tl.math.exp2((BT - tl.arange(0, BT)) * b_b)
    # [BT, BK]
    b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
    # [BS]
    d_q = tl.math.exp2(tl.arange(0, BS) * b_b)
    for i in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BS, BV), (1, 0))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * d_q[:, None]).to(b_do.dtype)

        b_dk *= d_b
        b_dv *= d_b
        # [BT, BS]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_s = tl.dot(b_kd, tl.trans(b_q), allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q, allow_tf32=False)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do, allow_tf32=False)
    b_dk *= d_h[:, None] * scale
    b_dv *= scale

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BS), tl.arange(0, BT)
    for i in range(i_t * BT, (i_t + 1) * BT, BS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BS, BV), (1, 0))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BS]
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) * b_b.to(tl.float32)), 0) * scale

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False) * d_s
        b_s = tl.dot(b_k, tl.trans(b_q), allow_tf32=False) * d_s
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q, allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        o_q += BS
    p_dk = tl.make_block_ptr(dk + (i_v * B * H + i_bh) * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_k * B * H + i_bh) * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV'])
})
@triton.jit
def parallel_retention_bwd_kernel(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
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
    i_h = i_bh % H
    parallel_retention_bwd_kernel_dq(
        i_bh,
        i_t,
        i_k,
        i_v,
        i_h,
        k,
        v,
        do,
        dq,
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
    parallel_retention_bwd_kernel_dkv(
        i_bh,
        i_t,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dk,
        dv,
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


def parallel_retention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    output_attentions: bool = False
):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT, BS = 64, 32
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

    grid = (NK * NV, triton.cdiv(T, BT), B * H)
    o = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None
    parallel_retention_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        attn=attn,
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
    return o, attn


def parallel_retention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT, BS = 64, 32
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
    grid = (NK * NV, triton.cdiv(T, BT), B * H)
    parallel_retention_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
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
    return dq, dk, dv


class ParallelRetentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, output_attentions):
        o, attn = parallel_retention_fwd(q, k, v, scale, output_attentions)
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return o.to(q.dtype), attn

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, d_attn=None):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = parallel_retention_bwd(q, k, v, do, ctx.scale)
        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None, None


def parallel_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    output_attentions: bool = False,
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
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, attn = ParallelRetentionFunction.apply(q, k, v, scale, output_attentions)
    if not head_first:
        o = o.transpose(1, 2)
    return o, attn

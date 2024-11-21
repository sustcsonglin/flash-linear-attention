# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None
})
@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # decay rate given the head index
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B*H + i_bh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B + i_b) * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_h = b_b * b_h + b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += K if HEAD_FIRST else H*K
        p_k += K if HEAD_FIRST else H*K
        p_v += V if HEAD_FIRST else H*V
        p_o += V if HEAD_FIRST else H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None
})
@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q,
    k,
    v,
    h0,
    do,
    dq,
    dk,
    dv,
    dh0,
    dht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK)
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B + i_b) * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_h = b_b * b_h + b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += K if HEAD_FIRST else H*K
        p_v += V if HEAD_FIRST else H*V
        p_do += V if HEAD_FIRST else H*V
        p_dq += K if HEAD_FIRST else H*K

    # sync threads
    tl.debug_barrier()

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_dk = dk + (i_bh + i_v * B*H) * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_dv = dv + (i_bh + i_k * B*H) * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_do = do + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_dk = dk + (i_v * B + i_b) * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_dv = dv + (i_k * B + i_b) * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_h, other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_dh *= b_b
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q -= K if HEAD_FIRST else H*K
        p_k -= K if HEAD_FIRST else H*K
        p_v -= V if HEAD_FIRST else H*V
        p_do -= V if HEAD_FIRST else H*V
        p_dk -= K if HEAD_FIRST else H*K
        p_dv -= V if HEAD_FIRST else H*V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def fused_recurrent_retention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape, dtype=torch.float)
    if output_final_state:
        final_state = q.new_empty(B, H, K, V, dtype=torch.float)
    else:
        final_state = None

    grid = (NV, NK, B*H)
    fused_recurrent_retention_fwd_kernel[grid](
        q, k, v, o, initial_state, final_state,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    o = o.sum(0)
    return o.to(v.dtype), final_state


def fused_recurrent_retention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_stages = 1
    num_warps = 1

    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)
    dh0 = q.new_empty(B, H, K, V, dtype=torch.float) if initial_state is not None else None

    grid = (NV, NK, B*H)
    fused_recurrent_retention_bwd_kernel[grid](
        q, k, v, initial_state, do, dq, dk, dv, dh0, dht,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    return dq, dk, dv, dh0


class FusedRecurrentRetentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, scale, initial_state=None, output_final_state=False, head_first=True):
        o, final_state = fused_recurrent_retention_fwd(q, k, v, scale, initial_state, output_final_state, head_first)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.head_first = head_first
        return o.to(v.dtype), final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, initial_state = ctx.saved_tensors
        scale = ctx.scale
        head_first = ctx.head_first
        dq, dk, dv, dh0 = fused_recurrent_retention_bwd(q, k, v, do, dht, scale, initial_state, head_first)
        return dq.to(q), dk.to(k), dv.to(v), None, dh0, None, None


def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
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
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRetentionFunction.apply(q, k, v, scale, initial_state, output_final_state, head_first)
    return o, final_state

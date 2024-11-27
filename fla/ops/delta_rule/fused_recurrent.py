# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.jit
def fused_recurrent_delta_rule_fwd_kernel(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, V]
    v,  # value [B, H, T, V].
    u,  # pseudo value [B, H, T, V]
    beta,  # beta [B, H, T]
    o,  # output [B, H, T, V]
    h0,
    ht,  # final hidden state [B, H, K, V]
    scale,  # K ** -0.5
    B,  # batch size
    T,  # seq_len
    H,  # n_heads
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    HEAD_FIRST: tl.constexpr,  # whether the inputs are in the head-first format
):

    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        p_u = u + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_bh * T
        p_o = o + (i_k * B*H + i_bh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_u = u + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_b * T*H + i_h
        p_o = o + (i_k * B + i_b) * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_u, b_v.to(p_v.dtype.element_ty), mask=mask_v)
        b_v *= b_beta
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += K if HEAD_FIRST else H*K
        p_k += K if HEAD_FIRST else H*K
        p_o += V if HEAD_FIRST else H*V
        p_v += V if HEAD_FIRST else H*V
        p_u += V if HEAD_FIRST else H*V
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit
def fused_recurrent_delta_rule_bwd_kernel(
    # NK: number of split in the K dimension
    # NV: number of split in the V dimension
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, V]
    v,  # value [B, H, T, V]
    beta,  # beta [B, H, T, (V)]
    h0,  # initial state [B, H, K, V]
    dh0,  # gradient of initial state [B, H, K, V]
    dht,  # gradient of final state [B, H, K, V]
    do,  # gradient of output [B, H, T, V]
    dq,  # gradient of query [NV, B, H, T, K]
    dk,  # gradient of key [NV, B, H, T, K]
    dv,  # gradient of value [NK, B, H, T, V]
    db,  # gradient of beta [NV, (NK), B, H, T]
    scale,  # K ** -0.5
    B: tl.constexpr,  # batch_size
    T: tl.constexpr,  # seq_len
    H: tl.constexpr,  # n_heads
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    NK: tl.constexpr,  # NK block size
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state h0
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar
    USE_DH0: tl.constexpr,  # whether to use dh0
    USE_DHT: tl.constexpr,  # whether to use dht
    HEAD_FIRST: tl.constexpr,  # whether the inputs are in the head-first format
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        p_dk = dk + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
        p_dv = dv + (i_k * B*H + i_bh) * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
        if IS_BETA_HEADWISE:
            p_beta = beta + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
            p_dbeta = db + (i_v * NK*B*H + i_k * B*H + i_bh) * T*V + tl.arange(0, BV) + (T - 1) * V
        else:
            p_beta = beta + i_bh * T + T - 1
            p_dbeta = db + (i_v * B*H + i_bh) * T + T - 1
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_do = do + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        p_dk = dk + (i_v * B + i_b) * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
        p_dv = dv + (i_k * B + i_b) * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
        if IS_BETA_HEADWISE:
            p_beta = beta + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
            p_dbeta = db + (i_v * NK*B + i_k * B + i_b) * T*H*V + i_h * V + tl.arange(0, BV) + (T - 1) * H*V
        else:
            p_beta = beta + i_b * T*H + (T - 1) * H + i_h
            p_dbeta = db + (i_v * B + i_b) * T*H + (T - 1) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * (b_v * b_beta)[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_db = b_dv * b_v if IS_BETA_HEADWISE else tl.sum(b_dv * b_v)
        b_dv = b_dv * b_beta

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        b_dh -= b_k[:, None] * b_dv[None, :]

        p_q -= K if HEAD_FIRST else H*K
        p_k -= K if HEAD_FIRST else H*K
        p_v -= V if HEAD_FIRST else H*V
        p_do -= V if HEAD_FIRST else H*V
        p_dk -= K if HEAD_FIRST else H*K
        p_dv -= V if HEAD_FIRST else H*V
        p_dbeta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)
        p_beta -= (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_bh * T
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B*H + i_bh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_b * T*H + i_h
        p_do = do + i_b * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B + i_b) * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dk = dk + (i_v * B + i_b) * T*H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B + i_b) * T*H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_dv = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk -= tl.sum(b_dv[None, :] * b_h, axis=1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += K if HEAD_FIRST else H*K
        p_v += V if HEAD_FIRST else H*V
        p_do += V if HEAD_FIRST else H*V
        p_dq += K if HEAD_FIRST else H*K
        p_dk += K if HEAD_FIRST else H*K
        p_dv += V if HEAD_FIRST else H*V
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    head_first: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, NK, B*H)
    u = torch.empty_like(v)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        u,
        beta,
        o,
        initial_state,
        final_state,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    head_first: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2

    beta_vector = beta.ndim == v.ndim

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, H, T, V) if head_first else q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, H, T) if head_first else q.new_empty(NV, B, T, H)
    grid = (NV, NK, B*H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    fused_recurrent_delta_rule_bwd_kernel[grid](
        q,
        k,
        v,
        beta,
        initial_state,
        dh0,
        dht,
        do,
        dq,
        dk,
        dv,
        db,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NK=NK,
        USE_INITIAL_STATE=initial_state is not None,
        USE_DH0=dh0 is not None,
        USE_DHT=dht is not None,
        IS_BETA_HEADWISE=beta_vector,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)

    return dq, dk, dv, db, dh0


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, beta, scale=None, initial_state=None, output_final_state=False, head_first=True):
        o, u, final_state = fused_recurrent_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            head_first=head_first
        )

        ctx.save_for_backward(q, k, u, beta, initial_state)
        ctx.scale = scale
        ctx.head_first = head_first
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, beta, initial_state = ctx.saved_tensors
        dq, dk, dv, db, dh0 = fused_recurrent_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            dht=dht,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            head_first=ctx.head_first
        )
        return dq.to(q), dk.to(k), dv.to(v), db.to(beta), None, dh0, None, None


def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
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
        beta (torch.Tensor):
             betas of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
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
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state, output_final_state, head_first)
    return o, final_state

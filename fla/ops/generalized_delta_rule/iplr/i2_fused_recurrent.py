# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


# on-the-fly computation without materializing hidden statets into HBMs
@triton.jit
def fused_recurrent_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V].
    alpha1,  # beta [B, H, L]
    beta1,
    alpha2,  # beta [B, H, L]
    beta2,
    o,   # output [B, H, L, V]
    ha1,  # tmp variable [B, H, L, V] for storing intermediate results of (h * alpha[None, :]).sum(0)
    ha2,  # tmp variable [B, H, L, V] for storing intermediate results of (h * alpha[None, :]).sum(0)
    h0,
    ht,  # final hidden state [B, H, K, V]
    s_qk_h,  # stride size: L * K
    s_vo_h,  # stride size: L * V
    scale,  # K ** -0.5
    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
):

    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha1 = alpha1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta1 = beta1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_alpha2 = alpha1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta2 = beta1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha1 = ha1 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha2 = ha2 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_alpha1 = tl.load(p_alpha1, mask=mask_bk, other=0).to(tl.float32)
        b_beta1 = tl.load(p_beta1, mask=mask_bk, other=0).to(tl.float32)
        b_alpha2 = tl.load(p_alpha1, mask=mask_bk, other=0).to(tl.float32)
        b_beta2 = tl.load(p_beta1, mask=mask_bk, other=0).to(tl.float32)
        # to store
        tmp1 = tl.sum(h * b_alpha1[None, :], axis=1)
        tmp2 = tl.sum(h * b_alpha2[None, :], axis=1)
        h += (tmp1[:, None] * b_beta1[None, :] + tmp2[:, None] * b_beta2[None, :] + b_k[None, :] * b_v[:, None])
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        tl.store(p_ha1, tmp1.to(p_ha1.dtype.element_ty), mask=mask_bv)
        tl.store(p_ha2, tmp2.to(p_ha2.dtype.element_ty), mask=mask_bv)
        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_ha1 += V
        p_alpha1 += K
        p_beta1 += K
        p_ha2 += V
        p_alpha2 += K
        p_beta2 += K

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    alpha1,  # alpha [B, H, L, K]
    beta1,  # beta [B, H, L, K]
    ha1,  # ha [B, H, L, V]
    alpha2,  # alpha [B, H, L, K]
    beta2,  # beta [B, H, L, K]
    ha2,  # ha [B, H, L, V]
    dht,  # gradient of final state [B, H, K, V]
    dh0,  # gradient of initial state [B, H, K, V]
    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    dalpha1,  # gradient of alpha [NV, B, H, L, K]
    dbeta1,  # gradient of beta [NV, B, H, L, K]
    dha1,  # gradient of ha [NK, B, H, L, V]
    dalpha2,  # gradient of alpha [NV, B, H, L, K]
    dbeta2,  # gradient of beta [NV, B, H, L, K]
    dha2,  # gradient of ha [NK, B, H, L, V]
    h0,  # initial state [B, H, K, V]
    s_qk_h,  # stride size: L * K
    s_vo_h,  # stride size: L * V
    NK,  # NK block size
    scale,  # K ** -0.5
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state h0
    USE_DH0: tl.constexpr,  # whether to use dh0
    USE_DHT: tl.constexpr,  # whether to use dht
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_ha1 = ha1 + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_alpha1 = alpha1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_beta1 = beta1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_ha2 = ha2 + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_alpha2 = alpha2 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_beta2 = beta2 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K

    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dbeta1 = dbeta1 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dha1 = dha1 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dbeta2 = dbeta2 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dha2 = dha2 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_beta1 = tl.load(p_beta1, mask=mask_bk, other=0).to(tl.float32)
        b_alpha1 = tl.load(p_alpha1, mask=mask_bk, other=0).to(tl.float32)
        b_ha1 = tl.load(p_ha1, mask=mask_bv, other=0).to(tl.float32)
        b_beta2 = tl.load(p_beta2, mask=mask_bk, other=0).to(tl.float32)
        b_alpha2 = tl.load(p_alpha2, mask=mask_bk, other=0).to(tl.float32)
        b_ha2 = tl.load(p_ha2, mask=mask_bv, other=0).to(tl.float32)

        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        b_dha1 = tl.sum(d_h * b_beta1[:, None], axis=0)
        b_dha2 = tl.sum(d_h * b_beta2[:, None], axis=0)
        tl.store(p_dha1, b_dha1.to(p_dha1.dtype.element_ty), mask=mask_bv)
        tl.store(p_dha2, b_dha2.to(p_dha2.dtype.element_ty), mask=mask_bv)
        b_dbeta1 = tl.sum(d_h * b_ha1[None, :], axis=1)
        b_dbeta2 = tl.sum(d_h * b_ha2[None, :], axis=1)
        tl.store(p_dbeta1, b_dbeta1.to(p_dbeta1.dtype.element_ty), mask=mask_bk)
        tl.store(p_dbeta2, b_dbeta2.to(p_dbeta2.dtype.element_ty), mask=mask_bk)

        d_h += b_dha1[None, :] * b_alpha1[:, None] + b_dha2[None, :] * b_alpha2[:, None]
        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V
        p_beta1 -= K
        p_dbeta1 -= K
        p_alpha1 -= K
        p_dha1 -= V
        p_ha1 -= V
        p_beta2 -= K
        p_dbeta2 -= K
        p_alpha2 -= K
        p_dha2 -= V
        p_ha2 -= V

    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h.to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta1 = beta1 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta2 = beta2 + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha1 = ha1 + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha2 = ha2 + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dha1 = dha1 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha1 = alpha1 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dalpha1 = dalpha1 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dha2 = dha2 + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha2 = alpha2 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dalpha2 = dalpha2 + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        d_ha1 = tl.load(p_dha1, mask=mask_bv, other=0).to(tl.float32)
        d_ha2 = tl.load(p_dha2, mask=mask_bv, other=0).to(tl.float32)
        d_alpha1 = tl.sum(d_ha1[None, :] * h, axis=1)
        d_alpha2 = tl.sum(d_ha2[None, :] * h, axis=1)
        tl.store(p_dalpha1, d_alpha1.to(p_dalpha1.dtype.element_ty), mask=mask_bk)
        tl.store(p_dalpha2, d_alpha2.to(p_dalpha2.dtype.element_ty), mask=mask_bk)
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_beta1 = tl.load(p_beta1, mask=mask_bk, other=0).to(tl.float32)
        b_beta2 = tl.load(p_beta2, mask=mask_bk, other=0).to(tl.float32)
        b_ha1 = tl.load(p_ha1, mask=mask_bv, other=0).to(tl.float32)
        b_ha2 = tl.load(p_ha2, mask=mask_bv, other=0).to(tl.float32)
        h += b_k[:, None] * b_v[None, :] + b_beta1[:, None] * b_ha1[None, :] + b_beta2[:, None] * b_ha2[None, :]
        _d_q = h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_dq += K
        p_k += K
        p_do += V
        p_v += V
        p_dk += K
        p_dalpha1 += K
        p_dha1 += V
        p_ha1 += V
        p_beta1 += K
        p_dalpha2 += K
        p_dha2 += V
        p_ha2 += V
        p_beta2 += K


class FusedRecurrentFunction2(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, alpha1, beta1, alpha2, beta2, scale=None, initial_state=None, output_final_state=False):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1
        assert NK == 1, "NK > 1 is not supported yet"
        o = q.new_empty(NK, B, H, T, V)

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None

        ha1 = torch.empty_like(v, dtype=torch.float32)
        ha2 = torch.empty_like(v, dtype=torch.float32)

        grid = (NV, NK, B * H)
        fused_recurrent_fwd_kernel[grid](
            q, k, v, alpha1, beta1, alpha2, beta2, o, ha1, ha2, initial_state, final_state,
            q.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        o = o.squeeze(0)
        ctx.save_for_backward(q, k, v, alpha1, beta1, ha1, alpha2, beta2, ha2, initial_state)
        ctx.scale = scale
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, alpha1, beta1, ha1, alpha2, beta2, ha2, initial_state = ctx.saved_tensors
        B, H, T, K, V = *q.shape, v.shape[-1]
        scale = ctx.scale
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        assert NK == 1, "NK > 1 is not supported yet"
        num_stages = 1
        num_warps = 2

        dq = q.new_empty(NV, B, H, T, K)
        dk = k.new_empty(NV, B, H, T, K)
        dv = v.new_empty(NK, B, H, T, V)

        dalpha1 = alpha1.new_empty(NV, B, H, T, K)
        dbeta1 = beta1.new_empty(NV, B, H, T, K)
        dha1 = ha1.new_empty(NK, B, H, T, V)
        dalpha2 = alpha1.new_empty(NV, B, H, T, K)
        dbeta2 = beta1.new_empty(NV, B, H, T, K)
        dha2 = ha1.new_empty(NK, B, H, T, V)

        grid = (NV, NK, B * H)

        if initial_state is not None and initial_state.requires_grad:
            dh0 = torch.empty_like(initial_state, dtype=torch.float32)
        else:
            dh0 = None

        fused_recurrent_bwd_kernel[grid](
            q, k, v, alpha1, beta1, ha1, alpha2, beta2, ha2, dht, dh0, do, dq, dk, dv,
            dalpha1, dbeta1, dha1, dalpha2, dbeta2, dha2, initial_state,
            q.stride(1),
            v.stride(1),
            NK, scale,
            B=B, H=H, T=T, K=K, V=V,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            USE_DH0=dh0 is not None,
            USE_DHT=dht is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        dalpha1 = dalpha1.sum(0)
        dbeta1 = dbeta1.sum(0)
        dalpha2 = dalpha2.sum(0)
        dbeta2 = dbeta2.sum(0)
        return (
            dq.to(q), dk.to(k), dv.to(v), dalpha1.to(alpha1), dbeta1.to(beta1),
            None, dalpha2.to(alpha2), dbeta2.to(beta2), dh0, None
        )


def fused_recurrent_iplr2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha1: torch.Tensor,
    beta1: torch.Tensor,
    alpha2: torch.Tensor,
    beta2: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence
    S_t = S_t @ (I + alpha_{1t} beta_{1t}^T + alpha_{2t} beta_{2t}^T) + v_t k_t^T
    in a recurrent manner.
    Since the transition matrices is identity-plus-2x-low-rank, we call it the Identity-Plus-Low-Rank2 (IPLR2).

    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        alpha1 (torch.Tensor):
            alphas of shape `(B, H, T, K)`
        beta2 (torch.Tensor):
             betas of shape `(B, H, T, K)`
        alpha2 (torch.Tensor):
            alphas of shape `(B, H, T, K)`
        beta2 (torch.Tensor):
             betas of shape `(B, H, T, K)`
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = FusedRecurrentFunction2.apply(
        q, k, v, alpha1, beta1, alpha2, beta2, scale, initial_state, output_final_state
    )
    return o, final_state

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.triton.utils import contiguous


@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    DK,
    DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, DV), (s_ht, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for _ in range(0, T, BT):
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)

        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (DK, 0))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    scale,
    DK,
    DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)

    for i_v in range(0, tl.cdiv(DV, BV)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (0, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, DV), (s_ht, 1), (i_t * DK, i_v * BV), (BK, BV), (1, 0))

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_s = tl.zeros([BT, BT], dtype=tl.float32)
        for _ in range(0, tl.cdiv(DK, BK)):
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [BD, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BD, BD]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
            b_s += tl.dot(b_q, b_k, allow_tf32=False)

            p_q = tl.advance(p_q, (0, BK))
            p_k = tl.advance(p_k, (BK, 0))
            p_h = tl.advance(p_h, (BK, 0))

        b_s *= d_s
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    do,
    dh,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_hh,
    s_ht,
    H,
    T,
    scale,
    DK,
    DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, ((i+1)*DK, DV), (s_ht, 1), (i * DK + i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)


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
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_hh,
    s_ht,
    H,
    T,
    TDK,
    scale,
    DK,
    DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    d_q = (d_q * scale).to(d_q.dtype)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale

    for i_k in range(0, tl.cdiv(DK, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (DV, TDK), (1, s_ht), (0, i_t * DK + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TDK, DV), (s_ht, 1), (i_t * DK + i_k * BK, 0), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))

        p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        for _ in range(tl.cdiv(DV, BV)):
            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            # [BK, BV]
            b_dh = tl.load(p_dh, boundary_check=(0, 1))

            # [BT, BT]
            b_ds = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
            b_ds = (b_ds * d_s).to(b_k.dtype)
            # [BT, BK]
            b_dq += tl.dot(b_do, b_h, allow_tf32=False) * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)

            # [BT, BT]
            b_ds = tl.trans(b_ds)
            # [BK, BT]
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None]
            b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
            b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
            b_dv += tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

            p_v = tl.advance(p_v, (0, BV))
            p_h = tl.advance(p_h, (BV, 0))
            p_do = tl.advance(p_do, (0, BV))
            p_dh = tl.advance(p_dh, (0, BV))
            p_dv = tl.advance(p_dv, (0, BV))

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, q, k, v, initial_state, output_final_state):
        BT = 64
        DK, DV = k.shape[-1], v.shape[-1]
        BK, BV = min(64, triton.next_power_of_2(DK)), min(64, triton.next_power_of_2(DV))
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = DK ** -0.5

        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        h = q.new_empty(batch_size, n_heads, triton.cdiv(seq_len, BT) * DK, DV)

        final_state = None
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, requires_grad=False)

        grid = (NK, NV, batch_size * n_heads)
        chunk_retention_fwd_kernel_h[grid](
            k, v, h, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2],
            DK=DK, DV=DV, BT=BT, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (triton.cdiv(seq_len, BT), batch_size * n_heads)
        o = torch.empty_like(v)
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2], scale,
            BK=BK, BV=BV, DK=DK, DV=DV, BT=BT,
            num_warps=num_warps,
            num_stages=num_stages
        )

        ctx.save_for_backward(q, k, v, h)
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do, d_ht=None):
        q, k, v, h = ctx.saved_tensors

        BT = 64
        DK, DV = k.shape[-1], v.shape[-1]
        BK, BV = min(64, triton.next_power_of_2(DK)), min(64, triton.next_power_of_2(DV))
        batch_size, n_heads, seq_len, _ = q.shape
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = DK ** -0.5

        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        grid = (NK, NV, batch_size * n_heads)
        dh = q.new_empty(batch_size, n_heads, triton.cdiv(seq_len, BT) * DK, DV)

        chunk_retention_bwd_kernel_dh[grid](
            q, do, dh,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            n_heads, seq_len, scale,
            BT=BT, BK=BK, BV=BV, DK=DK, DV=DV, NT=triton.cdiv(seq_len, BT),
            num_warps=num_warps,
            num_stages=num_stages
        )

        BK, BV = min(64, triton.next_power_of_2(DK)), min(64, triton.next_power_of_2(DV))
        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        grid = (triton.cdiv(seq_len, BT), batch_size * n_heads)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        # must be zero. we need reload
        dv = torch.zeros_like(v)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        chunk_retention_bwd_kernel_dqkv[grid](
            q, k, v, h, do, dh, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            n_heads, seq_len, h.shape[2], scale,
            BT=BT, BK=BK, BV=BV, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkRetentionFunction.apply(q, k, v, initial_state, output_final_state)
    if output_final_state:
        return o, final_state
    else:
        return o

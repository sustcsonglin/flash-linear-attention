# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous
from torch.cuda.amp import custom_bwd, custom_fwd


@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
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
    DK,
    DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TDK, DV),
                            (s_ht, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i) * b_b)
    # [DK, DV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(0, T, BT):
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [DK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [DK, DV]
        b_h = d_b * b_h + \
            tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (DK, 0))


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
    B,
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
    i_c, i_kv, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    NV = tl.cdiv(DV, BV)
    i_v = i_kv % NV
    i_k = i_kv // NV

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (i_c * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, i_c * BT), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (i_c * BT, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TDK, DV),
                            (s_ht, 1), (i_c * DK + i_k * BK, i_v * BV), (BK, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (i_c * BT, i_v * BV), (BT, BV), (1, 0))

    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2(o_i * b_b)
    # [BT, BD]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BD, BT]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BD]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BD, BD]
    b_h = tl.load(p_h, boundary_check=(0, 1))

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
    # [BT, BD]
    b_o = tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False) + \
        tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, do, dh,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_hh, s_ht,
    H, T, TDK,
    scale, DK, DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    p_dh = dh + i_bh * s_hh + (TDK - DK + i_k * BK + tl.arange(0, BK)
                               [:, None]) * DV + i_v * BV + tl.arange(0, BV)[None, :]
    mask = (i_k * BK + tl.arange(0, BK)
            [:, None] < DK) & (i_v * BV + tl.arange(0, BV)[None, :] < DV)
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2(o_i * b_b)
    # [DK, DV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range((tl.cdiv(T, BT) - 1) * BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T),
                                (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV),
                                 (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), mask=mask)
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [DK, DV]
        b_dh = d_b * b_dh + \
            tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)
        # p_dh = tl.advance(p_dh, (-DK, 0))
        p_dh -= DK * DV


@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h,
    do, dh, dq, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_hh, s_ht,
    B, H, T, TDK, scale, DK, DV,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_c, i_kv, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    NV = tl.cdiv(DV, BV)
    i_v = i_kv % NV
    i_k = i_kv // NV

    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (i_c * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, i_c * BT), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (i_c * BT, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (DV, TDK),
                            (1, s_ht), (i_v * BV, i_c * DK + i_k * BK), (BV, BK), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV),
                             (s_vo_t, s_vo_d), (i_c * BT, i_v * BV), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TDK, DV),
                             (s_ht, 1), (i_c * DK + i_k * BK, i_v * BV), (BK, BV), (1, 0))
    p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK),
                             (s_qk_t, s_qk_d), (i_c * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK),
                             (s_qk_t, s_qk_d), (i_c * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV),
                             (s_vo_t, s_vo_d), (i_c * BT, i_v * BV), (BT, BV), (1, 0))

    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)
    # [BD, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BD]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(
        p_v, boundary_check=(0, 1))
    # [BD, BD]
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dh = tl.load(p_dh, boundary_check=(0, 1))

    # [BT, BT]
    b_ds = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2(
        (o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    b_ds = (b_ds * d_s).to(b_k.dtype)
    # [BT, BD]
    b_dq = tl.dot((b_do * (d_q * scale)[:, None]).to(b_k.dtype),
                  b_h, allow_tf32=False) + tl.dot(b_ds, b_k, allow_tf32=False)

    # [BT, BT]
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_ds = tl.trans(b_ds)
    # [BT, BD]
    b_dk = tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * \
        d_k[:, None] + tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
    b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * \
        d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, q, k, v):
        BT = 64
        DK, DV = k.shape[-1], v.shape[-1]
        BK, BV = min(128, triton.next_power_of_2(DK)), min(
            128, triton.next_power_of_2(DV))
        batch_size, n_heads, seq_len, _ = q.shape
        num_stages = 3
        num_warps = 4
        scale = DK ** -0.5

        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        h = q.new_empty(batch_size, n_heads, triton.cdiv(seq_len, BT) * DK, DV)
        o = q.new_empty(NK, batch_size, n_heads, seq_len, DV)
        grid = (NK, NV, batch_size * n_heads)

        chunk_retention_fwd_kernel_h[grid](
            k, v, h,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2],
            DK=DK, DV=DV, BK=BK, BV=BV, BT=BT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (triton.cdiv(seq_len, BT), NK * NV, batch_size * n_heads)
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            batch_size, n_heads, seq_len, h.shape[2], scale,
            BK=BK, BV=BV, DK=DK, DV=DV, BT=BT,
            num_warps=num_warps,
            num_stages=num_stages
        )

        ctx.save_for_backward(q, k, v, h)
        return o.sum(0).to(q.dtype)

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do):
        q, k, v, h = ctx.saved_tensors

        BT = 64
        DK, DV = k.shape[-1], v.shape[-1]
        BK, BV = min(128, triton.next_power_of_2(DK)), min(
            128, triton.next_power_of_2(DV))
        batch_size, n_heads, seq_len, _ = q.shape
        num_stages = 3
        num_warps = 4
        scale = DK ** -0.5

        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        grid = (NK, NV, batch_size * n_heads)
        dh = q.new_empty(batch_size, n_heads,
                         triton.cdiv(seq_len, BT) * DK, DV)

        chunk_retention_bwd_kernel_dh[grid](
            q, do, dh,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            n_heads, seq_len, dh.shape[2], scale,
            BT=BT, BK=BK, BV=BV, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )

        BK, BV = min(64, triton.next_power_of_2(DK)), min(
            64, triton.next_power_of_2(DV))
        NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
        grid2 = (triton.cdiv(seq_len, BT), NK*NV, batch_size * n_heads)
        dq = q.new_empty(NV, batch_size, n_heads, seq_len, DK)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, DK)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, DV)
        num_stages = 3
        num_warps = 4
        chunk_retention_bwd_kernel_dqkv[grid2](
            q, k, v, h, do, dh, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            batch_size, n_heads, seq_len, h.shape[2], scale,
            BT=BT, BK=BK, BV=BV, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype)


chunk_retention = ChunkRetentionFunction.apply

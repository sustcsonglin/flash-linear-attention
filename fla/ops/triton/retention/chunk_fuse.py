# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

# on-the-fly computation without materializing hidden statets into HBMs


@triton.jit
def fused_chunk_retention_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))

    # d_b: overall decay for the entire chunk
    # d_o: cumulative decay from the start of the chunk
    # d_h: cumulative decay from the end of the chunk
    d_b, d_o, d_h = tl.math.exp2(BT * b_b), tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s

        # [BT, BV]
        b_o = tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # [BK, BV]
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_retention_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dk,  # gradient of key [NV, B, H, L, D_head_K]
    dv,  # gradient of value [NK, B, H, L, D_head_V]
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2(o_i * b_b) * scale, tl.math.exp2((BT - o_i) * b_b)
    d_b = tl.math.exp2(BT * b_b)

    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2(
        (o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i*BT, i_k*BK), (BT, BK), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [DV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)
        # [BT, DK]
        b_dq = tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False) + tl.dot(b_ds, b_k, allow_tf32=False)
        # [DV, DK]
        b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    d_s = tl.trans(d_s)
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i*BT, i_v*BV), (BT, BV), (1, 0))
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        # [BT, BT]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)

        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        # [BT, DK]
        b_dk = tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False) * d_k[:, None]
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, DV]
        b_dv = tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
        b_dv += tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        # [DK, DV]
        b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]

        scale = d_head_qk ** -0.5
        BT = 16
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        grid = (NV, NK, batch_size * n_heads)
        fused_chunk_retention_fwd_kernel[grid](
            q, k, v, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v)
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = d_head_qk ** -0.5

        BT = 16
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 2

        dq = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        fused_chunk_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv


fused_chunk_retention = FusedChunkRetentionFunction.apply

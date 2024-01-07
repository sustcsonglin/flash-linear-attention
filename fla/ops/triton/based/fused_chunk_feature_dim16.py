# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

from numpy import dtype
from regex import F
import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous

# on-the-fly computation without materializing hidden statets into HBMs


@triton.jit
def fused_chunk_based_fwd_kernel(
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

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV], zero-order taylor expansion
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    # [BK, BV], first-order taylor expansion
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_h_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k*B*H) * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0))
        # [BK*BK, BT]
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT]).to(b_k.dtype)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        # interchunk
        # zero-order
        b_o += b_h_0o
        # first-order
        b_o += tl.dot(b_q, b_h_1o.to(b_q.dtype), allow_tf32=False)
        # second-order
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK]).to(b_k.dtype)
        b_o += tl.dot(b_q_2o, b_h_2o.to(b_q_2o.dtype), allow_tf32=False) * 0.5

        # intrachunk
        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        # [TB, BV]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # update hidden state
        # [BK, BV]
        b_h_2o = b_h_2o + tl.dot(b_k_2o.to(b_v.dtype), b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_based_bwd_kernel(
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

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV], zero-order taylor expansion
    # b_h_0o = tl.zeros([BV], dtype=tl.float32)
    # [BK, BV], first-order taylor expansion
    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_h_2o = tl.zeros([BV, BK*BK], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(
            k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(
            v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * s_qk_h,
                                 (T, DK), (s_qk_t, s_qk_d), (i*BT, i_k*BK), (BT, BK), (1, 0))

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)

        # load tensors
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0))
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0))
        b_k = tl.load(p_k, boundary_check=(0))
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(1))

        # inter-chunk
        b_dq += tl.dot(b_do, (b_h_1o).to(b_do.dtype), allow_tf32=False)
        b_dq_2o = tl.dot(b_do, (b_h_2o).to(b_do.dtype), allow_tf32=False) * 0.5
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale

        # intra-chunk
        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * scale
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((b_ds * (1 + b_s)).to(b_q.dtype), b_k, allow_tf32=False)

        # store
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0))

        # update hidden state
        # [BT, BK*BK]
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)
        # [BV, BK*BK]
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o.to(b_v.dtype), allow_tf32=False)
        # [BV, BK]
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)

    tl.debug_barrier()
    b_h_1o = None
    b_h_2o = None

    # [BK, BV], first-order taylor expansion
    b_dh_1o = tl.zeros([BK, BV], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_dh_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)
    b_dh_0o = tl.zeros([BV], dtype=tl.float32)

    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]

    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(
            k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(
            v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * s_qk_h, (T, DK),
                                 (s_qk_t, s_qk_d), (T - i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * s_vo_h, (T, DV),
                                 (s_vo_t, s_vo_d), (T - i*BT, i_v*BV), (BT, BV), (1, 0))

        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)

        # intra chunk
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds *= (1+b_s)

        b_dk += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2.to(b_do.dtype), b_do, allow_tf32=False)

        # inter chunk
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)

        b_dv += tl.dot(b_k, b_dh_1o.to(b_k.dtype), allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o.to(b_k.dtype), allow_tf32=False)
        b_dv += b_dh_0o

        b_dk += tl.dot(b_v, tl.trans(b_dh_1o).to(b_k.dtype), allow_tf32=False)
        b_dk_2o = tl.dot(b_dh_2o.to(b_k.dtype),
                         tl.trans(b_v), allow_tf32=False)
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BV])
        b_k_fp32 = tl.trans(b_k.to(tl.float32))
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)

        # hidden state update
        b_dh_0o += tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT]).to(b_k.dtype)
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkBasedFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        assert d_head_qk == 16, "currently we do not support feature dim other than 16"
        d_head_v = v.shape[-1]

        scale = d_head_qk ** -0.5
        BT = 16
        BK, BV = min(d_head_qk, 16), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_warps = 4

        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        grid = (NV, NK, batch_size * n_heads)
        fused_chunk_based_fwd_kernel[grid](
            q, k, v, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v)
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = d_head_qk ** -0.5

        BT = 16
        BK, BV = min(d_head_qk, 16), min(d_head_v, 16)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 4

        dq = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        fused_chunk_based_bwd_kernel[grid](
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


fused_chunk_based_dim16 = FusedChunkBasedFunction.apply

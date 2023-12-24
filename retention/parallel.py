# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl


@triton.jit
def parallel_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    b,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)
    b_b = tl.load(p_b)
    # [BQ, BD]
    b_q = tl.load(p_q)
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BQ, BD], dtype=tl.float32)
    for _ in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_k = tl.load(p_k)
        # [BK, BD]
        b_v = tl.load(p_v)

        # [BQ, BK]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BQ, BD]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BK))
        p_v = tl.advance(p_v, (BK, 0))
        o_k += BK
    tl.store(p_o, b_o.to(p_o.dtype.element_ty))


@triton.jit
def parallel_retention_bwd_kernel_dq(
    k,
    v,
    b,
    do,
    dq,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)
    b_b = tl.load(p_b)
    # [BQ, BD]
    b_do = tl.load(p_do)
    # [BQ, BD]
    b_dq = tl.zeros([BQ, BD], dtype=tl.float32)
    for _ in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_k = tl.load(p_k)
        # [BD, BK]
        b_v = tl.load(p_v)

        # [BQ, BK]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0) * scale
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s
        # [BQ, BD]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BK, 0))
        p_v = tl.advance(p_v, (0, BK))
        o_k += BK
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))


@triton.jit
def parallel_retention_bwd_kernel_dkv(
    q,
    k,
    v,
    b,
    do,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_k * BK, tl.arange(0, BK) + i_k * BK
    b_b = tl.load(p_b)
    # [BK, BD]
    b_k, b_v = tl.load(p_k), tl.load(p_v)
    # [BK, BD]
    b_dk, b_dv = tl.zeros([BK, BD], dtype=tl.float32), tl.zeros([BK, BD], dtype=tl.float32)
    for i in range(i_k * BK, T, BQ):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i), (BD, BQ), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i), (BD, BQ), (0, 1))

        # [BD, BQ]
        b_q = tl.load(p_q)
        b_do = tl.load(p_do)

        # [BK, BQ]
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) * b_b.to(tl.float32)), 0) * scale
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * d_s

        # [BK, BD]
        b_dk += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        o_q += BQ
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))


class ParallelRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BQ, BK, BD = 64, 64, triton.next_power_of_2(q.shape[-1])
        batch_size, n_heads, seq_len, d_head = q.shape
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)
        scale = d_head ** -0.5

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))

        o = torch.empty_like(q)
        # NOTE: be careful about BF16 precision
        b = (1. - q.new_tensor(2., dtype=torch.float).pow(-5 - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        parallel_retention_fwd_kernel[grid](
            q, k, v, o, b,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, b)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    def backward(ctx, do):
        q, k, v, b = ctx.saved_tensors
        scale = ctx.scale
        BD = triton.next_power_of_2(q.shape[-1])
        batch_size, n_heads, seq_len, d_head = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, q.shape)

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        BQ, BK = 64, 64
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)
        parallel_retention_bwd_kernel_dq[grid](
            k, v, b, do, dq,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        BK, BQ = 64, 64
        grid = (triton.cdiv(seq_len, BK), batch_size * n_heads)
        parallel_retention_bwd_kernel_dkv[grid](
            q, k, v, b, do, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head]


parallel_retention = ParallelRetentionFunction.apply

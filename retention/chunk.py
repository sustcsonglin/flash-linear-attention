# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl


@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    b,
    s_qh,
    s_qt,
    s_qd,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    BT: tl.constexpr,
    BD: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (i_k * DK, 0), (DK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, i_v * DV), (BT, DV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, BD), (s_ht, s_qd), (i_k * DK, i_v * DV), (DK, DV), (1, 0))
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i) * b_b)
    # [DK, DV]
    b_h = tl.zeros([DK, DV], dtype=tl.float32)
    for _ in range(0, T, BT):
        tl.store(p_h, b_h.to(p_h.dtype.element_ty))

        # [DK, BT]
        b_k = tl.load(p_k)
        # [BT, DV]
        b_v = tl.load(p_v)
        # [DK, DV]
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), False)

        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (BD, 0))


@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    b,
    s_qh,
    s_qt,
    s_qd,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i_q * BT), (BD, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, BD), (s_ht, s_qd), (i_q * BD, 0), (BD, BD), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_i = tl.math.exp2(o_i * b_b)
    # [BT, BD]
    b_q = tl.load(p_q)
    b_q = (b_q * scale).to(b_q.dtype)
    # [BD, BT]
    b_k = tl.load(p_k)
    # [BT, BD]
    b_v = tl.load(p_v)
    # [BD, BD]
    b_h = tl.load(p_h)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_s = tl.dot(b_q, b_k, False) * d_s
    # [BT, BD]
    b_o = tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, False) + tl.dot(b_s.to(b_q.dtype), b_v, False)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty))


@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    b,
    do,
    dh,
    s_qh,
    s_qt,
    s_qd,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (i_k * DK, T - BT), (DK, BT), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - BT, i_v * DV), (BT, DV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TD, BD), (s_ht, s_qd), (TD - BD + i_k * DK, i_v * DV), (DK, DV), (1, 0))
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2(o_i * b_b)
    # [DK, DV]
    b_dh = tl.zeros([DK, DV], dtype=tl.float32)
    for _ in range(0, T, BT):
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty))

        # [DK, BT]
        b_q = tl.load(p_q)
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, DV]
        b_do = tl.load(p_do)
        # [DK, DV]
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), False)

        p_q = tl.advance(p_q, (0, -BT))
        p_do = tl.advance(p_do, (-BT, 0))
        p_dh = tl.advance(p_dh, (-BD, 0))


@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q,
    k,
    v,
    h,
    b,
    do,
    dh,
    dq,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    s_hh,
    s_ht,
    H,
    T,
    TD,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i_q * BT), (BD, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (BD, TD), (s_qd, s_ht), (0, i_q * BD), (BD, BD), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TD, BD), (s_ht, s_qd), (i_q * BD, 0), (BD, BD), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BT, 0), (BT, BD), (1, 0))
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_q, d_k = tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)
    # [BD, BT]
    b_q = tl.load(p_q)
    # [BT, BD]
    b_do = tl.load(p_do)
    b_k, b_v = tl.load(p_k), tl.load(p_v)
    # [BD, BD]
    b_h = tl.load(p_h)
    b_dh = tl.load(p_dh)

    # [BT, BT]
    b_ds = tl.dot(b_do, tl.trans(b_v), False)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    b_ds = (b_ds * d_s).to(b_k.dtype)
    # [BT, BD]
    b_dq = tl.dot((b_do * (d_q * scale)[:, None]).to(b_k.dtype), b_h, False) + tl.dot(b_ds, b_k, False)

    # [BT, BT]
    b_s = tl.dot(b_k, b_q, False) * tl.trans(d_s)
    b_ds = tl.trans(b_ds)
    # [BT, BD]
    b_dk = tl.dot(b_v, tl.trans(b_dh), False) * d_k[:, None] + tl.dot(b_ds, tl.trans(b_q), False)
    b_dv = tl.dot(b_k, b_dh, False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, False)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BD = triton.next_power_of_2(q.shape[-1])
        BT = 32 if BD > 64 else 64
        DK, DV = 64, 64
        batch_size, n_heads, seq_len, d_head = q.shape
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4
        scale = d_head ** -0.5
        assert seq_len % BT == 0, f"seq_len {seq_len} must be divisible by block_size {BT}"

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))

        h = q.new_empty(batch_size, n_heads, triton.cdiv(seq_len, BT) * BD, BD)
        o = torch.empty_like(q)
        # NOTE: be careful about BF16 precision
        b = (1. - q.new_tensor(2., dtype=torch.float).pow(-5 - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        grid = (triton.cdiv(BD, DK), triton.cdiv(BD, DV), batch_size * n_heads)
        chunk_retention_fwd_kernel_h[grid](
            k, v, h, b,
            q.stride(1), q.stride(2), q.stride(3), h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2],
            BT=BT, BD=BD, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (triton.cdiv(seq_len, BT), batch_size * n_heads)
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o, b,
            q.stride(1), q.stride(2), q.stride(3), h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2], scale,
            BT=BT, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, h, b)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    def backward(ctx, do):
        q, k, v, h, b = ctx.saved_tensors
        scale = ctx.scale
        BD = triton.next_power_of_2(q.shape[-1])
        BT = 32 if BD > 64 else 64
        DK, DV = 64, 64
        batch_size, n_heads, seq_len, d_head = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, q.shape)

        dq, dk, dv, dh = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(h)
        grid = (triton.cdiv(BD, DK), triton.cdiv(BD, DV), batch_size * n_heads)
        chunk_retention_bwd_kernel_dh[grid](
            q, b, do, dh,
            q.stride(1), q.stride(2), q.stride(3), h.stride(1), h.stride(2),
            n_heads, seq_len, h.shape[2], scale,
            BT=BT, BD=BD, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (triton.cdiv(seq_len, BT), batch_size * n_heads)
        chunk_retention_bwd_kernel_dqkv[grid](
            q, k, v, h, b, do, dh, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3), dh.stride(1), dh.stride(2),
            n_heads, seq_len, h.shape[2], scale,
            BT=BT, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head]


chunk_retention = ChunkRetentionFunction.apply

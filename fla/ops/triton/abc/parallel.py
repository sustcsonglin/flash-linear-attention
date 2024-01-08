# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous


@triton.jit
def flash_abc_fwd_kernel(
    q,
    k,
    v,
    ek,
    ev,
    zk,
    zv,
    s,
    p,
    o,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (0, 0), (BK, BM), (1, 0))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, 0), (BM, BK), (0, 1))
    p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)

    # [BQ, BD]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BQ, BD], dtype=tl.float32)
    # [BQ, BM]
    b_s = tl.zeros([BQ, BM], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_k = tl.load(p_k)
        # [BK, BM]
        b_ek = tl.load(p_ek)

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        b_qk = tl.dot(b_q, b_k, allow_tf32=False)
        b_qk = tl.where(m_qk, b_qk, 0).to(b_q.dtype)
        # [BQ, BM]
        b_s += tl.dot(b_qk, b_ek, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BK))
        p_ek = tl.advance(p_ek, (BK, 0))
    b_s = b_s / tl.load(p_zk)
    b_z = tl.exp(b_s - tl.max(b_s, 1)[:, None])
    b_p = tl.fdiv(b_z, tl.sum(b_z, 1)[:, None])

    tl.store(p_s, b_s.to(p_s.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    # [BQ, BM]
    b_p = (b_p / tl.load(p_zv, boundary_check=(0, 1))).to(b_q.dtype)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BM, BK]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BD]
        b_o += tl.dot(tl.where(m_qk, tl.dot(b_p, b_ev, allow_tf32=False), 0).to(b_v.dtype), b_v, allow_tf32=False)

        p_v = tl.advance(p_v, (BK, 0))
        p_ev = tl.advance(p_ev, (0, BK))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_abc_bwd_kernel_dqsp(
    k,
    v,
    ek,
    ev,
    zv,
    p,
    do,
    doo,
    dq,
    ds,
    dp,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (0, 0), (BK, BM), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)

    # [BQ, BD]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BQ, BM]
    b_dp = tl.zeros([BQ, BM], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BM]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BM]
        b_dp += tl.dot(tl.where(m_qk, tl.dot(b_do, b_v, allow_tf32=False), 0).to(b_v.dtype), b_ev, allow_tf32=False)

        p_v = tl.advance(p_v, (0, BK))
        p_ev = tl.advance(p_ev, (BK, 0))

    tl.debug_barrier()

    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, 0), (BM, BK), (0, 1))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_doo = tl.make_block_ptr(doo + i_bh * T, (T,), (s_qd,), (i_q * BQ,), (BQ,), (0,))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))

    # [BQ, BM]
    b_p = tl.load(p_p, boundary_check=(0, 1))
    b_zv = tl.load(p_zv, boundary_check=(0, 1))
    b_dp = (b_dp / b_zv).to(b_p.dtype)
    # [BQ,]
    b_doo = tl.load(p_doo, boundary_check=(0,))
    # [BQ, BM]
    b_ds = (b_p * (b_dp - b_doo[:, None]) * scale).to(b_p.dtype)
    # [BQ, BD]
    b_dq = tl.zeros([BQ, BD], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM, BK]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BD]
        b_dq += tl.dot(tl.where(m_qk, tl.dot(b_ds, b_ek, allow_tf32=False), 0).to(b_k.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BK, 0))
        p_ek = tl.advance(p_ek, (0, BK))
    tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_abc_bwd_kernel_dkv(
    q,
    k,
    v,
    ek,
    ev,
    s,
    p,
    do,
    ds,
    dp,
    dk,
    dv,
    dsk,
    dsv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dsk = tl.make_block_ptr(dsk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_dsv = tl.make_block_ptr(dsv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))

    o_q, o_k = tl.arange(0, BQ), tl.arange(0, BK) + i_k * BK

    # [BK, BD]
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    # [BK, BM]
    b_ek, b_ev = tl.load(p_ek, boundary_check=(0, 1)), tl.load(p_ev, boundary_check=(0, 1))
    # [BK, BD]
    b_dk, b_dv = tl.zeros([BK, BD], dtype=tl.float32), tl.zeros([BK, BD], dtype=tl.float32)
    # [BK, BM]
    b_dsk, b_dsv = tl.zeros([BK, BM], dtype=tl.float32), tl.zeros([BK, BM], dtype=tl.float32)

    for i in range(i_k * BK, T, BQ):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i, 0), (BQ, BD), (1, 0))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i, 0), (BQ, BD), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))

        # [BQ, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BQ, BM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        # [BQ, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BQ, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))

        # [BK, BQ]
        m_kq = o_k[:, None] <= (o_q + i)[None, :]

        bed = tl.where(m_kq, tl.dot(b_ek, tl.trans(b_ds), allow_tf32=False), 0.).to(b_do.dtype)
        # [BK, BQ]
        b_dk += tl.dot(bed, b_q, allow_tf32=False)
        # [BK, BQ]
        b_kq = tl.where(m_kq, tl.dot(b_k, tl.trans(b_q), allow_tf32=False), 0.).to(b_do.dtype)
        b_dsk += tl.dot(b_kq, b_ds, allow_tf32=False)
        b_dsk -= tl.dot(tl.where(m_kq, 1., 0.).to(b_do.dtype), b_ds * b_s, allow_tf32=False)

        b_ep = tl.where(m_kq, tl.dot(b_ev, tl.trans(b_p), allow_tf32=False), 0.).to(b_do.dtype)
        # [BK, BD]
        b_dv += tl.dot(b_ep, b_do, allow_tf32=False)
        # [BK, BQ]
        b_vdo = tl.where(m_kq, tl.dot(b_v, tl.trans(b_do), allow_tf32=False), 0.).to(b_do.dtype)
        b_dsv += tl.dot(b_vdo, b_p, allow_tf32=False)
        b_dsv -= tl.dot(tl.where(m_kq, 1., 0.).to(b_do.dtype), b_dp * b_p, allow_tf32=False)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dsk, (b_ek * b_dsk).to(p_dsk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dsv, (b_ev * b_dsv).to(p_dsv.dtype.element_ty), boundary_check=(0, 1))


class ParallelABCAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        BQ, BK, BD, BM = 64, 64, triton.next_power_of_2(q.shape[-1]), triton.next_power_of_2(sk.shape[-1])
        batch_size, n_heads, seq_len, d_head, n_slots = *q.shape, sk.shape[-1]
        scale = d_head ** -0.5
        num_stages = 1
        num_warps = 4
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))
        s = torch.empty_like(sk)
        p = torch.empty_like(s)
        o = torch.empty_like(q)
        sk, sv = (sk - sk.max(2, True)[0]).float(), (sv - sv.max(2, True)[0]).float()
        zk, zv = sk.logcumsumexp(2), sv.logcumsumexp(2)
        ek, zk = (sk - zk[:, :, -1:]).to(q.dtype).exp(), (zk - zk[:, :, -1:]).to(q.dtype).exp()
        ev, zv = (sv - zv[:, :, -1:]).to(q.dtype).exp(), (zv - zv[:, :, -1:]).to(q.dtype).exp()

        flash_abc_fwd_kernel[grid](
            q * scale, k, v, ek, ev, zk, zv, s, p, o,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, ek, ev, zk, zv, s, p, o)
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.n_slots = n_slots
        ctx.grid = grid
        ctx.dtype = q.dtype
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, ek, ev, zk, zv, s, p, o = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale

        BQ, BK, BD, BM = 64, 64, q.shape[-1], ek.shape[-1]
        seq_len, d_head, n_slots = ctx.seq_len, ctx.d_head, ctx.n_slots
        num_stages = 1
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, o.shape)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        ds, dp, dsk, dsv = torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek)

        doo = (o * do).sum(-1)
        flash_abc_bwd_kernel_dqsp[grid](
            k, v, ek, ev, zv, p / zk, do, doo, dq, ds, dp,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len, scale,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        flash_abc_bwd_kernel_dkv[grid](
            q, k, v, ek, ev, s / scale, p / zv, do, ds, dp, dk, dv, dsk, dsv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head], dsk[..., :n_slots], dsv[..., :n_slots]


parallel_abc_attention = ParallelABCAttentionFunction.apply

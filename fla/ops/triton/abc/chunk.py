# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous


@triton.jit
def chunk_abc_fwd_kernel_s(
    q,
    k,
    s,
    ek,
    zk,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BD, DM]
    b_hk = tl.zeros([BD, DM], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BT, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BD, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DM]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_zk = tl.load(p_zk, boundary_check=(0, 1))

        # [BT, DM]
        b_s = tl.dot(b_q, b_hk.to(b_q.dtype), allow_tf32=False)
        b_s += tl.dot(tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False), 0).to(b_q.dtype), b_ek, allow_tf32=False)
        b_s = b_s / b_zk
        # [BD, DM]
        b_hk += tl.dot(b_k, b_ek, allow_tf32=False)

        tl.store(p_s, b_s.to(p_s.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_o(
    v,
    o,
    p,
    ev,
    zv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DV: tl.constexpr
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, DV]
    b_hv = tl.zeros([BM, DV], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, i * BT), (BM, BT), (0, 1))
        p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))

        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        # [BM, BT]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        # [BT, BM]
        b_zv = tl.load(p_zv, boundary_check=(0, 1))

        b_p = (b_p / b_zv).to(b_v.dtype)
        # [BT, DV]
        b_o = tl.dot(b_p, b_hv.to(b_v.dtype), allow_tf32=False)
        b_o += tl.dot(tl.where(m_s, tl.dot(b_p, b_ev, allow_tf32=False), 0).to(b_v.dtype), b_v, allow_tf32=False)
        # [BM, DV]
        b_hv += tl.dot(b_ev, b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dsp(
    v,
    p,
    ev,
    zk,
    zv,
    do,
    doo,
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
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BD, DM]
    b_hv = tl.zeros([BD, DM], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_doo = tl.make_block_ptr(doo + i_bh * T, (T,), (s_qd,), (i * BT,), (BT,), (0,))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BD, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        b_zk = tl.load(p_zk, boundary_check=(0, 1))
        b_zv = tl.load(p_zv, boundary_check=(0, 1))
        # [BT, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT,]
        b_doo = tl.load(p_doo, boundary_check=(0,))

        # [BT, DM]
        b_dp = tl.dot(b_do, b_hv.to(b_do.dtype), allow_tf32=False)
        b_dp += tl.dot(tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False), 0).to(b_v.dtype), b_ev, allow_tf32=False)
        b_dp = (b_dp / b_zv).to(b_p.dtype)
        b_ds = (b_p / b_zk * (b_dp - b_doo[:, None]) * scale).to(b_p.dtype)
        # [BD, DM]
        b_hv += tl.dot(b_v, b_ev, allow_tf32=False)

        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dq(
    k,
    ek,
    dq,
    ds,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, DK]
    b_hk = tl.zeros([BM, DK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, i * BT), (BM, BT), (0, 1))
        p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM, BT]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        # [BT, BM]
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, DK]
        b_dq = tl.dot(b_ds, b_hk.to(b_k.dtype), allow_tf32=False)
        b_dq += tl.dot(tl.where(m_s, tl.dot(b_ds, b_ek, allow_tf32=False), 0).to(b_k.dtype), b_k, allow_tf32=False)
        # [BM, DK]
        b_hk += tl.dot(b_ek, b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dkv(
    q,
    p,
    ek,
    ev,
    do,
    ds,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] <= o_i[None, :]

    # [BM, DK]
    b_dhk = tl.zeros([BM, DK], dtype=tl.float32)
    b_dhv = tl.zeros([BM, DK], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, T - i * BT), (BM, BT), (0, 1))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, 0), (BT, BM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, 0), (BT, BM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, T - i * BT), (BM, BT), (0, 1))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))

        # [BT, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BM, BT]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, BT]
        b_dk = tl.dot(b_ek, b_dhk.to(b_ek.dtype), allow_tf32=False)
        b_dk += tl.dot(tl.where(m_s, tl.dot(b_ek, b_ds, allow_tf32=False), 0.).to(b_do.dtype), b_q, allow_tf32=False)

        # [BT, BD]
        b_dv = tl.dot(b_ev, b_dhv.to(b_ev.dtype), allow_tf32=False)
        b_dv += tl.dot(tl.where(m_s, tl.dot(b_ev, b_p, allow_tf32=False), 0.).to(b_do.dtype), b_do, allow_tf32=False)
        # [BM, DK]
        b_dhk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dhv += tl.dot(b_p, b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dskv(
    q,
    k,
    v,
    s,
    p,
    ek,
    ev,
    do,
    ds,
    dp,
    dsk,
    dsv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] <= o_i[None, :]

    # [BD, DM]
    b_dhk = tl.zeros([BD, DM], dtype=tl.float32)
    b_dhv = tl.zeros([BD, DM], dtype=tl.float32)
    # [DM,]
    b_zdss, b_zdpp = tl.zeros([DM,], dtype=tl.float32), tl.zeros([DM,], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, T - i * BT), (BD, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, 0), (BT, BD), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, 0), (BT, BD), (1, 0))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, T - i * BT), (BD, BT), (0, 1))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dsk = tl.make_block_ptr(dsk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dsv = tl.make_block_ptr(dsv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BT, BD]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [BD, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, DM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))

        # [BT, DM]
        b_dss = b_ds * b_s
        # [BT, BT]
        b_dsk = tl.dot(tl.where(m_s, tl.dot(b_k, b_q, allow_tf32=False), 0.).to(b_k.dtype), b_ds, allow_tf32=False)
        b_dsk -= tl.dot(tl.where(m_s, 1., 0.).to(b_k.dtype), b_dss.to(b_k.dtype), allow_tf32=False)
        b_dsk += tl.dot(b_k, b_dhk.to(b_k.dtype), allow_tf32=False) - b_zdss[None, :]

        # [BT, DM]
        b_dpp = b_dp * b_p
        # [BT, BT]
        b_dsv = tl.dot(tl.where(m_s, tl.dot(b_v, b_do, allow_tf32=False), 0.).to(b_v.dtype), b_p, allow_tf32=False)
        b_dsv -= tl.dot(tl.where(m_s, 1., 0.).to(b_v.dtype), b_dpp.to(b_v.dtype), allow_tf32=False)
        b_dsv += tl.dot(b_v, b_dhv.to(b_v.dtype), allow_tf32=False) - b_zdpp[None, :]
        # [BD, DM]
        b_dhk += tl.dot(b_q, b_ds, allow_tf32=False)
        b_dhv += tl.dot(b_do, b_p, allow_tf32=False)
        # [DM,]
        b_zdss += tl.sum(b_dss, 0)
        b_zdpp += tl.sum(b_dpp, 0)

        tl.store(p_dsk, (b_ek * b_dsk).to(p_dsk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dsv, (b_ev * b_dsv).to(p_dsv.dtype.element_ty), boundary_check=(0, 1))


class ChunkABCAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        batch_size, n_heads, seq_len, d_head, n_slots = *q.shape, sk.shape[-1]
        scale = d_head ** -0.5
        BT, BD, BM = 16, triton.next_power_of_2(q.shape[-1]), triton.next_power_of_2(sk.shape[-1])
        DV, DM = min(BD, 32), min(BM, 32)
        NV, NM = triton.cdiv(BD, DV), triton.cdiv(BM, DM)
        num_stages = 1
        num_warps = 2

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))
        o = torch.empty_like(q)
        s = torch.empty_like(sk)
        sk, sv = sk.float(), sv.float()
        zk, zv = sk.logcumsumexp(2), sv.logcumsumexp(2)
        ek, ev = (sk - zk[:, :, -1:]).exp().to(q.dtype), (sv - zv[:, :, -1:]).exp().to(q.dtype)
        zk, zv = (zk - zk[:, :, -1:]).exp().to(q.dtype), (zv - zv[:, :, -1:]).exp().to(q.dtype)

        grid = (NM, batch_size * n_heads)
        chunk_abc_fwd_kernel_s[grid](
            q * scale, k, s, ek, zk,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        p = s.softmax(-1, dtype=torch.float).to(q.dtype)
        grid = (NV, batch_size * n_heads)
        chunk_abc_fwd_kernel_o[grid](
            v, o, p, ev, zv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, o, s, p, ek, ev, zk, zv)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.n_slots = n_slots
        ctx.dtype = q.dtype
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, p, ek, ev, zk, zv = ctx.saved_tensors
        BT, BD, BM = 16, q.shape[-1], ek.shape[-1]
        DK, DM = min(BD, 64), min(BM, 32)
        NK, NM = triton.cdiv(BD, DK), triton.cdiv(BM, DM)
        batch_size, n_heads, seq_len, d_head, n_slots = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head, ctx.n_slots
        scale = ctx.scale
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
        grid = (NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dsp[grid](
            v, p, ev, zk, zv, do, doo, ds, dp,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len, scale,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NK, batch_size * n_heads)
        chunk_abc_bwd_kernel_dq[grid](
            k, ek, dq, ds,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DK=DK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        s, p = s / scale, p / zv
        chunk_abc_bwd_kernel_dkv[grid](
            q, p, ek, ev, do, ds, dk, dv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DK=DK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dskv[grid](
            q, k, v, s, p, ek, ev, do, ds, dp, dsk, dsv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head], dsk[..., :n_slots], dsv[..., :n_slots]


chunk_abc_attention = ChunkABCAttentionFunction.apply

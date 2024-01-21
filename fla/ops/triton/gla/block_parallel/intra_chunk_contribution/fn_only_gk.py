# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.triton.utils import contiguous


@triton.jit
def _fwd_kernel_compute_A(
    Q,
    K,
    GK,
    A,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_q4,
    stride_a1,
    stride_a2,
    stride_a3,
    stride_a4,
    Z,
    H,
    N_CTX,
    D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)

    qk_offset = off_hz * stride_q2 + off_k * BLOCK_DMODEL_QK
    a_offset = (off_k * Z*H + off_hz) * stride_a2

    lo = 0
    hi = BLOCK_N

    Q_ptr = Q + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        :, None] + tl.arange(0, 16)[None, :] * stride_q4

    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(
        0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4

    GK_Q_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(
        0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0,
                                                             16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    for q_high in range(16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 +
                               q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2.to(q.dtype)

        # inter-chunk bf16
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4).to(tl.float32)
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk.to(k.dtype)
            qk = tl.dot(q, k, allow_tf32=False)
            tl.store(A_ptr + q_high * stride_a4 + k_high,
                     qk.to(A_ptr.dtype.element_ty))

    # intra chunk fp32
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 +
                               q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k = k * tl.trans(q_gk3)

        qk = tl.dot(q, k, allow_tf32=False)
        qk = tl.where(tl.arange(0, 16)[:, None]
                      >= tl.arange(0, 16)[None, :], qk, 0.)
        tl.store(A_ptr + q_high * stride_a4 + q_high,
                 qk.to(A_ptr.dtype.element_ty))


@triton.jit
def _bwd_kernel_dqk(
    Q,
    K,
    GK,
    DA,
    DQ,
    DK,
    DGK,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_q4,
    stride_a1,
    stride_a2,
    stride_a3,
    stride_a4,
    Z,
    H,
    N_CTX,
    D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)

    qk_offset = off_hz * stride_q2 + BLOCK_DMODEL_QK * off_k
    a_offset = off_hz * stride_a2

    lo = 0
    hi = BLOCK_N

    Q_ptr = Q + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(
        0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    GK_Q_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(
        0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    # DGK_Q_ptr = DGK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DA_ptr = DA + a_offset + (start_m) * stride_a3 + tl.arange(0,
                                                               16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    # inter chunk dq. bf16
    for q_high in range(lo+16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)

        q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3) +
                               q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK)).to(tl.float32)

        # q2 = q * q_gk.to(q.dtype)

        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4).to(tl.float32)
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high).to(k.dtype)
            k_gk = tl.exp(q_normalizer[None, :] - k_gk)
            k = k * k_gk.to(k.dtype)
            dq2 += tl.dot(dqk, k, allow_tf32=False)

        dq2 = dq2.to(q.dtype)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        dq = dq2 * q_gk.to(q.dtype)
        dq_gk = dq * q

        DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
            None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        tl.store(DQ_ptr, dq.to(DQ_ptr.dtype.element_ty))

        DGK_Q_ptr = DGK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
            None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        # prev = tl.load(DGK_Q_ptr)
        tl.store(DGK_Q_ptr, dq_gk.to(DGK_Q_ptr.dtype.element_ty))

    tl.debug_barrier()

    for k_high in range(lo, hi-16, 16):
        k = tl.load(K_ptr + k_high * stride_q4)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dgk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for q_high in range(k_high+16, hi, 16):
            q = tl.load(Q_ptr + q_high * stride_q4)
            q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3) + q_high * stride_q4 + tl.arange(0,
                                                                                                           BLOCK_DMODEL_QK)).to(tl.float32)
            q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
            q_gk = tl.exp(q_gk - q_normalizer[None, :]).to(q.dtype)
            q = q * q_gk
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high).to(q.dtype)

            k_gk2 = tl.exp(q_normalizer[None, :] - k_gk)

            dk2 = tl.dot(tl.trans(dqk), q, allow_tf32=False)
            dk += dk2 * k_gk2
            dgk -= dk2 * k * k_gk2

        DK_ptr = DK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
            None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        tl.store(DK_ptr, dk.to(DK_ptr.dtype.element_ty))

        DGK_K_ptr = DGK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
            None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        prev = tl.load(DGK_K_ptr)
        tl.store(DGK_K_ptr,  (prev + dgk).to(DGK_K_ptr.dtype.element_ty))

    tl.debug_barrier()

    DK_ptr = DK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DGK_K_ptr = DGK + qk_offset + (start_m) * stride_q3 + tl.arange(
        0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[
        None, :] + tl.arange(0, 16)[:, None] * stride_q4

    # intra chunk, fp32.
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 +
                               q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q2 = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)

        k = tl.load(K_ptr + q_high * stride_q4)
        k2 = k * q_gk3

        dqk = tl.load(DA_ptr + q_high * stride_a4 + q_high)
        dqk = tl.where(tl.arange(0, 16)[:, None]
                       >= tl.arange(0, 16)[None, :], dqk, 0.)

        dk2 = tl.dot(tl.trans(dqk), q2, allow_tf32=False)
        dk = dk2 * q_gk3
        prev_dk = tl.load(DK_ptr + q_high * stride_q4)
        tl.store(DK_ptr + q_high * stride_q4,
                 (dk + prev_dk).to(DK_ptr.dtype.element_ty))

        dgk = - dk * k
        dq2 = tl.dot(dqk, k2, allow_tf32=False)
        dq = dq2 * q_gk2

        prev_dq = tl.load(DQ_ptr + q_high * stride_q4)
        tl.store(DQ_ptr + q_high * stride_q4,
                 (dq + prev_dq).to(DQ_ptr.dtype.element_ty))

        dgk += dq * q
        prev_dq_gk = tl.load(DGK_K_ptr + q_high * stride_q4)
        tl.store(DGK_K_ptr + q_high * stride_q4,
                 (dgk + prev_dq_gk).to(DGK_K_ptr.dtype.element_ty))


class IntraCalA(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, q, k, gk):

        # assert gk.dtype==torch.float32
        # only support for Ampere now

        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80")

        # assert gk.dtype == gv.dtype == torch.float32
        # for now.
        BLOCK_M = BLOCK_N = q.shape[-2]

        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        if Lk > 128:
            assert Lk % 128 == 0

        BLOCK_DMODEL_QK = min(Lk, 128)
        ctx.BLOCK_DMODEL_QK = BLOCK_DMODEL_QK

        A = torch.zeros(max(1, Lk//128), q.shape[0], q.shape[1],
                        q.shape[2], BLOCK_N, BLOCK_N, device=q.device, dtype=q.dtype)

        grid = (q.shape[2], q.shape[0] * q.shape[1], max(1, Lk//128))

        # assert q.dtype == k.dtype == v.dtype
        _fwd_kernel_compute_A[grid](
            q, k, gk, A,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            # be careful here!
            A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=BLOCK_DMODEL_QK, BLOCK_M=BLOCK_M, num_warps=8 if ctx.BLOCK_DMODEL_QK == 128 else 4, num_stages=8
        )

        ctx.save_for_backward(q, k, gk)
        ctx.grid = grid
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_N = BLOCK_N
        ctx.head = q.shape[1]
        return A.sum(0).to(q.dtype)

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, dA):
        q, k,  gk = ctx.saved_tensors

        # appearantly, there is no sync issue when splitting K dim.
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dgk = torch.zeros_like(gk)

        BLOCK_N = ctx.BLOCK_N
        # for now.
        BLOCK_M = BLOCK_N

        _bwd_kernel_dqk[ctx.grid](
            q, k, gk, dA,
            dq,
            dk, dgk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            dA.stride(0), dA.stride(1), dA.stride(2), dA.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_QK=ctx.BLOCK_DMODEL_QK,
            BLOCK_M=BLOCK_M,
            num_warps=8 if ctx.BLOCK_DMODEL_QK == 128 else 4,
            num_stages=5
        )

        return dq.to(q.dtype), dk.to(k.dtype), dgk.to(gk.dtype)

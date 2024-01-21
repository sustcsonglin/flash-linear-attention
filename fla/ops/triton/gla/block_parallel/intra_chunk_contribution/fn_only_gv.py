# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.triton.utils import contiguous


@triton.jit
def _fwd_compute_O(
    A,
    V,
    GV,
    O,
    stride_a2,
    stride_a3,
    stride_a4,
    stride_v2,
    stride_v3,
    stride_v4,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)

    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V

    lo = 0
    hi = BLOCK_N

    V_ptr = V + v_offset + (start_m) * stride_v3 + tl.arange(0,
                                                             BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    O_ptr = O + v_offset + (start_m) * stride_v3 + tl.arange(0,
                                                             BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    GV_ptr = GV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0,
                                                             16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    for q_high in range(lo+16, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 +
                                  q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)

        # k_gv = tl.load(GV_ptr + q_high * stride_v4)
        # q_gv = tl.exp(k_gv - q_gv_normalizer[None, :])

        for k_high in range(0, q_high, 16):
            qk = tl.load(A_ptr + q_high * stride_a4 + k_high)
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v = v * k_gv.to(v.dtype)
            # bf16
            output = tl.dot(qk.to(v.dtype), v, allow_tf32=False)
            acc += output

        tl.store(O_ptr + q_high * stride_v4, acc.to(O.dtype.element_ty))

    tl.store(O_ptr, tl.zeros([16, BLOCK_DMODEL_V],
             dtype=tl.float32).to(O.dtype.element_ty))

    tl.debug_barrier()

    for q_high in range(lo, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 +
                                  q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        qk = tl.load(A_ptr + q_high * stride_a4 + q_high)
        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)

        # fp32 matmul
        v = v * k_gv2
        output = tl.dot(qk.to(tl.float32), v, allow_tf32=False)

        q_gv = tl.exp(k_gv - q_gv_normalizer[None, :])

        prev = tl.load(O_ptr + q_high * stride_v4)
        output += prev
        output = output * q_gv

        tl.store(O_ptr + q_high * stride_v4, output.to(O.dtype.element_ty))


@triton.jit
def _bwd_kernel_dav(
    V,
    GV,
    A,
    O,
    DO,
    DA,
    DV,
    DGV,
    Z,
    H,
    stride_a1,
    stride_a2,
    stride_a3,
    stride_a4,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_v4,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr
):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)

    a_offset = off_hz * stride_a2
    da_offset = (off_v * Z * H + off_hz) * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V

    lo = 0
    hi = BLOCK_N

    DO_ptr = DO + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4

    O_ptr = O + v_offset + (start_m) * stride_v3 + tl.arange(0,
                                                             BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    DV_ptr = DV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4

    GV_ptr = GV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4

    DGV_ptr = DGV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0,
                                                             16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    DA_ptr = DA + da_offset + (start_m) * stride_a3 + tl.arange(0,
                                                                16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    # pre-compute do*q_gv. in-place update
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        o = tl.load(O_ptr + q_high * stride_v4)
        tl.store(DGV_ptr + q_high * stride_v4, (do * o))

        q_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 +
                                  q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
        q_gv = tl.load(GV_ptr + q_high * stride_v4)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        do = do * q_gv

        tl.store(DO_ptr + q_high * stride_v4, do.to(DO_ptr.dtype.element_ty))

    tl.debug_barrier()

    V_ptr = V + v_offset + (start_m) * stride_v3 + \
        tl.arange(0, BLOCK_DMODEL_V)[:, None] + tl.arange(0, 16)[None, :] * stride_v4
    GV_ptr = GV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        :, None] + tl.arange(0, 16)[None, :] * stride_v4

    for q_high in range(lo+16, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        q_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 + q_high *
                                  stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        for k_high in range(0, q_high, 16):
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[:, None] - k_gv)

            # bf16
            v2 = v * k_gv.to(v.dtype)
            dqk = tl.dot(do, v2, allow_tf32=False)
            tl.store(DA_ptr + q_high * stride_a4 +
                     k_high, dqk.to(DA.dtype.element_ty))

    tl.debug_barrier()

    A_ptr = A + a_offset + (start_m) * stride_a3 + \
        tl.arange(0, 16)[:, None] + tl.arange(0, 16)[None, :] * stride_a4

    V_ptr = V + v_offset + (start_m) * stride_v3 + \
        tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + (start_m) * stride_v3 + \
        tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    for k_high in range(0, hi, 16):
        dv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)

        k_gv = tl.load(GV_ptr + k_high * stride_v4)

        for q_high in range(k_high + 16, BLOCK_N, 16):
            do = tl.load(DO_ptr + q_high * stride_v4)

            kq = tl.load(A_ptr + q_high * stride_a4 + k_high).to(do.dtype)

            q_gv_normalizer = tl.load(GV + v_offset +
                                      start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
            k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)

            # bf16
            dv2 = tl.dot(kq, do, allow_tf32=False)
            dv += dv2 * k_gv2

        v = tl.load(V_ptr + k_high * stride_v4)
        tl.store(DV_ptr + k_high * stride_v4, dv.to(v.dtype))

        prev_dv = tl.load(DGV_ptr + k_high * stride_v4)
        tl.store(DGV_ptr + k_high * stride_v4, prev_dv - dv*v)

    tl.debug_barrier()

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0,
                                                             16)[:, None] + tl.arange(0, 16)[None, :] * stride_a4

    # intra-chunk
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)

        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 +
                                  q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v2 = v * k_gv

        dqk = tl.dot(do.to(v2.dtype), tl.trans(v2), allow_tf32=False)
        dqk = tl.where(tl.arange(0, 16)[:, None]
                       >= tl.arange(0, 16)[None, :], dqk, 0.)
        tl.store(DA_ptr + q_high * stride_a4 + q_high,
                 dqk.to(DA_ptr.dtype.element_ty))

        kq = tl.load(A_ptr + q_high * stride_a4 + q_high).to(do.dtype)
        dv2 = tl.dot(kq, do, allow_tf32=False)

        dv = dv2 * k_gv
        prev_dv = tl.load(DV_ptr + q_high * stride_v4)
        tl.store(DV_ptr + q_high * stride_v4,
                 (prev_dv + dv).to(DV.dtype.element_ty))

        prev_gdv = tl.load(DGV_ptr + q_high * stride_v4)
        prev_gdv -= dv * v
        tl.store(DGV_ptr + q_high * stride_v4,
                 prev_gdv.to(DGV.dtype.element_ty))


class IntraCalO(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, A, v, gv):
        assert gv.dtype == torch.float32
        # assert A.dtype == torch.float32

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80")

        # assert gk.dtype == gv.dtype == torch.float32
        BLOCK_M = BLOCK_N = v.shape[-2]

        # shape constraints
        Lv = v.shape[-1]
        BLOCK_V = min(128, Lv)
        ctx.BLOCK_V = BLOCK_V

        assert v.shape[-1] % BLOCK_V == 0

        grid = (v.shape[2], v.shape[0] * v.shape[1],
                max(1, v.shape[-1] // BLOCK_V))

        o = torch.empty_like(v)

        _fwd_compute_O[grid](A, v, gv, o,
                             A.stride(0), A.stride(
                                 1), A.stride(2), A.stride(3),
                             v.stride(0), v.stride(
                                 1), v.stride(2), v.stride(3),
                             BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                             BLOCK_DMODEL_V=BLOCK_V, num_warps=8 if BLOCK_V == 128 else 4, num_stages=5
                             )

        ctx.save_for_backward(A, v, gv, o)
        ctx.grid = grid
        return o

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do):
        A, v,  gv, o = ctx.saved_tensors
        BLOCK_V = ctx.BLOCK_V
        assert v.shape[-1] % BLOCK_V == 0

        # dA = torch.empty_like(A)
        dv = torch.zeros_like(v)
        dgv = torch.zeros_like(gv)

        # for now.
        BLOCK_M = BLOCK_N = v.shape[-2]

        # shape constraints
        # Lv = v.shape[-1]
        # grid = (v.shape[2] , v.shape[0] * v.shape[1],  v.shape[-1] // BLOCK_V)
        grid = ctx.grid

        dA = torch.empty(v.shape[-1] // BLOCK_V if BLOCK_V == 128 else 1, A.shape[0],
                         A.shape[1], A.shape[2], A.shape[3], A.shape[3], device=A.device, dtype=A.dtype)

        _bwd_kernel_dav[grid](
            v, gv, A, o,
            do, dA,
            dv, dgv,
            v.shape[0], v.shape[1],
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_V=ctx.BLOCK_V, num_warps=8, num_stages=4
        )

        return dA.sum(0).to(A), dv.to(v), dgv.to(gv)

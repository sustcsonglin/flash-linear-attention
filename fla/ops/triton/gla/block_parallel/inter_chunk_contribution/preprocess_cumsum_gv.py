# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from fla.ops.triton.utils import contiguous


@triton.jit
def _fwd_preprocess_cumsum_gv(
    V,
    GV,
    GV_cumsum,
    GV_exp,
    V_reduce,
    GV_last_exp,
    NUM_CHUNK,
    L,
    D_MODEL_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)

    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    GV_last_exp_ptr = GV_last_exp + offset_bh * NUM_CHUNK * \
        D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)

    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_exp_ptr = GV_exp + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    cumsum = tl.zeros([D_MODEL_V], dtype=tl.float32)

    for _ in range(CHUNK_SIZE):
        gv = tl.load(GV_ptr).to(tl.float32)
        cumsum += gv

        tl.store(GV_cumsum_ptr, cumsum.to(GV_cumsum_ptr.dtype.element_ty))
        tl.store(GV_exp_ptr, tl.exp(cumsum).to(GV_cumsum_ptr.dtype.element_ty))

        GV_cumsum_ptr += D_MODEL_V
        GV_exp_ptr += D_MODEL_V
        GV_ptr += D_MODEL_V

    tl.store(GV_last_exp_ptr, tl.exp(cumsum).to(
        GV_last_exp_ptr.dtype.element_ty))

    tl.debug_barrier()

    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    V_reduce_ptr = V_reduce + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    for _ in range(CHUNK_SIZE):
        v = tl.load(V_ptr)
        gv = tl.load(GV_cumsum_ptr)
        v_reduce = v * tl.exp(cumsum - gv)
        tl.store(V_reduce_ptr, v_reduce.to(V_reduce_ptr.dtype.element_ty))

        V_ptr += D_MODEL_V
        V_reduce_ptr += D_MODEL_V
        GV_cumsum_ptr += D_MODEL_V


@triton.jit
def _bwd_preprocess_cumsum_gv(
    V,
    GV,
    GV_cumsum,
    DGV_cumsum_exp,
    DV_reduce,
    DGV_last_exp,
    DGV_cumsum,
    DV,
    DGV,
    NUM_CHUNK,
    L,
    D_MODEL_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DV_ptr = DV + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DV_reduce_ptr = DV_reduce + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_ptr = DGV_cumsum + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_exp_ptr = DGV_cumsum_exp + offset_bh * L * D_MODEL_V + \
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * \
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    D_GV_last_exp_ptr = DGV_last_exp + offset_bh * NUM_CHUNK * \
        D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)

    cumsum_gradient = tl.zeros([D_MODEL_V], dtype=tl.float32)
    grad_gv_last = tl.zeros([D_MODEL_V], dtype=tl.float32)

    gv_last = tl.load(GV_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_V)
    cumsum_gradient += tl.load(D_GV_last_exp_ptr) * \
        tl.exp(gv_last).to(tl.float32)

    GV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    GV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    V_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    DV_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gv_cs = tl.load(GV_cumsum_ptr).to(tl.float32)
        v = tl.load(V_ptr).to(tl.float32)
        grad_v = tl.exp(gv_last - gv_cs) * \
            tl.load(DV_reduce_ptr).to(tl.float32)
        tl.store(DV_ptr, grad_v.to(DV_ptr.dtype.element_ty))
        grad_v *= v
        cumsum_gradient -= grad_v
        grad_gv_last += grad_v

        # q = tl.load(Q_ptr).to(tl.float32)
        grad_v = tl.exp(gv_cs) * tl.load(DGV_cumsum_exp_ptr)
        cumsum_gradient += grad_v

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGV_cumsum_ptr).to(tl.float32)

        tl.store(DGV_ptr, cumsum_gradient.to(DGV_ptr.dtype.element_ty))

        V_ptr -= D_MODEL_V
        DV_reduce_ptr -= D_MODEL_V
        GV_cumsum_ptr -= D_MODEL_V
        DGV_cumsum_ptr -= D_MODEL_V
        DV_ptr -= D_MODEL_V
        DGV_ptr -= D_MODEL_V
        DGV_cumsum_exp_ptr -= D_MODEL_V

    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * \
        D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * \
        D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V

    grad_gv_last = grad_gv_last + 0.

    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgv = tl.load(DGV_ptr).to(tl.float32)
        dgv += grad_gv_last
        tl.store(DGV_ptr, dgv.to(DGV_ptr.dtype.element_ty))
        DGV_ptr -= D_MODEL_V
        GV_ptr -= D_MODEL_V


class PreprocessCumSum_GV(torch.autograd.Function):
    @staticmethod
    @contiguous
    @torch.cuda.amp.custom_fwd
    def forward(ctx, v, gv):
        B, H, NUM_CHUNK, CHUNK_SIZE, D_v = v.shape

        grid = (B * H, NUM_CHUNK)
        ctx.grid = grid

        gv_cumsum = torch.empty_like(gv, dtype=torch.float32)
        gv_cumsum_exp = torch.empty_like(gv)
        v_reduce = torch.empty_like(v)
        gv_last_exp = torch.empty_like(gv[:, :, :, 0], dtype=torch.float32)
        _fwd_preprocess_cumsum_gv[grid](
            v, gv,  gv_cumsum, gv_cumsum_exp,
            v_reduce, gv_last_exp,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK=NUM_CHUNK, L=CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_V=D_v, num_warps=8 if D_v >= 512 else 4
        )

        ctx.grid = grid
        ctx.save_for_backward(v, gv, gv_cumsum)
        return gv_cumsum, v_reduce, gv_cumsum_exp, gv_last_exp

    @staticmethod
    @contiguous
    def backward(ctx, dgv_cumsum, dv_reduce, dgv_cumsum_exp, dgv_last_exp):
        v, gv, gv_cumsum = ctx.saved_tensors
        grid = ctx.grid

        B, H, NUM_CHUNK, CHUNK_SIZE, D_v = v.shape

        dv = torch.empty_like(v)
        dgv = torch.empty_like(gv)
        _bwd_preprocess_cumsum_gv[grid](
            v, gv, gv_cumsum,  dgv_cumsum_exp, dv_reduce, dgv_last_exp, dgv_cumsum,
            dv, dgv,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK=NUM_CHUNK, L=CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_V=D_v, num_warps=8 if D_v >= 512 else 4
        )
        return dv.to(v.dtype), dgv.to(gv.dtype)

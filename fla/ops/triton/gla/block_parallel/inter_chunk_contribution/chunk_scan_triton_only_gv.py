# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.triton.utils import contiguous


@triton.jit
def _fwd_recurrence(
    S,
    p2,
    O,
    NUM_BLOCK,
    D_MODEL_K: tl.constexpr,
    D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + \
        tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * \
        BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + \
        tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + \
        tl.arange(0, BLOCK_MODEL)[None, :] + D_MODEL_K * D_MODEL_V

    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + \
        tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + D_MODEL_V

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)

    S += D_MODEL_K * D_MODEL_V

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V

    for i in range(NUM_BLOCK-2):
        p_v = tl.load(p2)
        S_i = tl.load(S)
        acc = acc * p_v[None, :] + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        p2 += D_MODEL_V
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V


# NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL
@triton.jit
def _bwd_recurrence(
    S,
    p2,
    DS,
    Dp2,
    NUM_BLOCK,
    NUM_SPLIT_K,
    NUM_SPLIT_V,
    D_MODEL_K: tl.constexpr,
    D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr

):

    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + \
        tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(
            0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + \
        tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(
            0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    # skip the last chunk because it is never used
    # p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K

    # skip the last chunk because it is never used
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + \
        tl.arange(0, BLOCK_MODEL) + offset_s * \
        BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V

    # skip the last chunk because it is never used
    # NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V: stride_bh
    # offset_s * D_MODEL_K: find the right split in the K dimension
    # Dp1 = Dp1 + offset_bh * NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V + offset_s * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K * NUM_SPLIT_V

    # skip the last chunk because it is never used
    Dp2 = Dp2 + offset_bh * NUM_BLOCK * D_MODEL_V * NUM_SPLIT_K + offset_d * D_MODEL_V + \
        tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + \
        (NUM_BLOCK - 2) * D_MODEL_V * NUM_SPLIT_K

    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):

        # p_key = tl.load(p1)
        p_value = tl.load(p2)
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        Dacc += DS_i
        dp_i = Dacc * S_i
        # dp_key = tl.sum(dp_i * p_value[None, :], axis=1)
        # tl.store(Dp1, dp_key.to(Dp1.dtype.element_ty))
        dp_value = tl.sum(dp_i, axis=0)
        tl.store(Dp2, dp_value.to(Dp2.dtype.element_ty))

        tl.store(S, Dacc.to(S.dtype.element_ty))

        # Dacc *= p_key[:, None]
        Dacc *= p_value[None, :]

        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        # p1 -= D_MODEL_K
        p2 -= D_MODEL_V
        # Dp1 -= D_MODEL_K * NUM_SPLIT_V
        Dp2 -= D_MODEL_V * NUM_SPLIT_K


class Chunk_memory_update_only_gv(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx,  decay_value_last, to_add):
        B, H, N, D_k, D_v = to_add.shape
        output = torch.empty_like(to_add)
        BLOCK_MODEL = 32

        assert D_k % 32 == 0
        assert D_v % 32 == 0
        # assert D_k == decay_key_last.shape[-1]
        assert D_v == decay_value_last.shape[-1]

        grid = (B*H, D_k//BLOCK_MODEL, D_v//BLOCK_MODEL)
        ctx.grid = grid
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            to_add,
            decay_value_last,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N,
            BLOCK_MODEL=BLOCK_MODEL
        )

        output[:, :, 0] = 0
        ctx.save_for_backward(output,  decay_value_last)

        return output

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, DO):

        output, decay_value_last = ctx.saved_tensors

        B, H, N, D_k, D_v = output.shape

        num_block = N

        BLOCK_MODEL = 32

        grid = (B*H, D_k//BLOCK_MODEL, D_v//BLOCK_MODEL)

        # I don't want atomic_add to be used in the backward pass
        # so I add another dimension to the output tensor (D_k/v // BLOCK_MODEL)
        # afterward, I sum over this dimension to get the correct gradient
        D_p2 = torch.empty(B, H, N, D_k // BLOCK_MODEL, D_v,
                           device=DO.device, dtype=torch.float32)

        _bwd_recurrence[grid](
            output,  decay_value_last,
            DO, D_p2,
            NUM_BLOCK=num_block, NUM_SPLIT_K=D_k // BLOCK_MODEL, NUM_SPLIT_V=D_v // BLOCK_MODEL,
            D_MODEL_K=D_k,
            D_MODEL_V=D_v,
            BLOCK_MODEL=BLOCK_MODEL
        )

        output[:, :, -1] = 0
        # D_p1[:, :, 0] = 0
        # D_p1[:, :, -1] = 0
        D_p2[:, :, 0] = 0
        D_p2[:, :, -1] = 0

        return D_p2.sum(-2).to(decay_value_last.dtype), output

# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang
# Gated Linear Attention Transformers with Hardware-Efficient Training: https://arxiv.org/abs/2312.06635
# on-the-fly computation without materializing hidden statets into HBMs

import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from fla.ops.triton.utils import contiguous, require_version
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import semiring_cal_A
except ImportError:
    warnings.warn('Failed to import semiring_cal_A. Do not use FusedChunk implementation of GLA.')

inv_ln2 = 1.44269504


@triton.jit
def fused_chunk_gla_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    g,  # cumulative sum of log decay [B, H, L, D_head_K]
    o,  # output [B, H, L, D_head_V]

    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]

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
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h,
                            (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV,
                                (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_g *= inv_ln2

        d_b = tl.load(p_db) * inv_ln2

        b_q = (b_q * scale * tl.math.exp2(b_g))
        b_k = b_k * tl.trans(tl.math.exp2(-b_g + d_b[None, :]))

        b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
        b_h *= tl.math.exp2(d_b)[:, None]
        b_h += tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * DK

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(
            final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty),
                 boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_gla_bwd_kernel(
    q, k, v, g,
    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dk,  # gradient of key [NV, B, H, L, D_head_K]
    dv,  # gradient of value [NK, B, H, L, D_head_V]

    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]

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
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV,
                                (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(
            k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(
            g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + \
            ((i+1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(
            v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh+i_v*B*H)*s_qk_h, (T, DK),
                                 (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        d_b = tl.load(p_db) * inv_ln2

        # [DV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
        # [DV, DK]
        b_k *= tl.math.exp2(d_b[None, :] - b_g)
        b_h *= tl.math.exp2(d_b)[None, :]
        b_h += tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        b_dq *= scale * tl.math.exp2(b_g)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    # cum = tl.zeros([BK], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(
            k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(
            g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + \
            (T - (i-1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(
            v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK),
                                 (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        # p_dg = tl.make_block_ptr(dg + (i_bh + i_v * B * H) * s_qk_h, (T, DK),
        #  (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV),
                                 (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        b_db = tl.load(p_db) * inv_ln2

        # inter-chunk
        g_k = tl.math.exp2(b_db[None, :] - b_g)
        b_k *= g_k
        b_q *= tl.math.exp2(tl.trans(b_g))
        b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(
            b_v), allow_tf32=False)) * scale * g_k
        b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(
            b_v.dtype), allow_tf32=False) * scale

        # [DK, DV]
        b_dh *= tl.math.exp2(b_db)[:, None]
        b_dh += tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        ctx.scale = scale

        # inter-chunk
        BT = 16  # chunk_size
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        num_stages = 1
        num_warps = 2

        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        g = rearrange(g, 'b h (n c) d -> b h n c d', c=BT)
        g = g.float().cumsum(-2)
        g = rearrange(g, 'b h n c d -> b h (n c) d')

        if output_final_state:
            final_state = q.new_empty(
                batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None

        grid = (NV, NK, batch_size * n_heads)
        fused_chunk_gla_fwd_kernel[grid](
            q, k, v, g, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            # clamp_min=-3,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            # USE_SIGMOID=True, USE_EXP=False,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)

        # ### intra-chunk
        chunk_size = 16
        num_chunk = seq_len // chunk_size
        q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
        k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
        A = semiring_cal_A.forward(q2, k2, g2) * scale
        o2 = A @ v2
        o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
        o.add_(o2)
        ctx.save_for_backward(q, k, v, g, A, initial_state)
        return o.to(v), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, g, A, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale

        # inter-chunk
        BT = 16
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 2

        dq = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads,  seq_len, d_head_v)

        grid = (NV, NK, batch_size * n_heads)
        fused_chunk_gla_bwd_kernel[grid](
            q, k, v, g, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            # clamp_min=-3,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)

        dg = dq * q
        dg.add_(- dk * k)

        # # # #### intra chunk
        num_chunk = seq_len // BT
        q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
        k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=num_chunk)
        dA2 = (do2 @ v2.transpose(-2, -1)) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dq2, dk2, dg2 = semiring_cal_A.backward(q2, k2, g2, dA2)
        dq2 = rearrange(dq2, '... h n c d -> ... h (n c) d')
        dk2 = rearrange(dk2, '... h n c d -> ... h (n c) d')
        dv2 = rearrange(dv2, '... h n c d -> ... h (n c) d')
        dg2 = rearrange(dg2, '... h n c d -> ... h (n c) d')
        dq.add_(dq2.to(dq))
        dk.add_(dk2.to(dk))
        dv.add_(dv2.to(dv))
        dg = dg.float()
        dg.add_(dg2)
        dg_cumsum = dg.cumsum(-2)
        dg = dg - dg_cumsum + dg_cumsum[:, :, -1, None]
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), None, None, None


def pad(x, chunk_size=16):
    seq_len = x.shape[-2]
    padded_seq_len = ceildiv(seq_len, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - seq_len))
    if x.shape[-1] % 32 != 0:
        x = F.pad(x, (0, 32 - x.shape[-1] % 32))
    return x


def ceildiv(a, b):
    return -(a // -b)


@require_version('triton>=2.2', 'Numerical stability consideration!')
def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    seq_len = v.shape[-2]
    d_head_v = v.shape[-1]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(
        q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :d_head_v]
    if output_final_state:
        return o, final_state
    return o

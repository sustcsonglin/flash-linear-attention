# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang

# Gated Linear Attention Transformers with Hardware-Efficient Training: https://arxiv.org/abs/2312.06635
# on-the-fly computation without materializing hidden statets into HBMs

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.utils import contiguous

inv_ln2 = 1.44269504


@triton.jit
def safe_exy(x, y):
    # e^x * y = sign(y) * e^(x + log(|y|)
    # this utility is designed for the safe multiplication of two variables e^x and y,
    # where x is in log space and y in normal space respectively.
    # it is important to ensure that e^x * y will not result in an overflow
    return tl.where(y > 0, 1., -1.) * tl.exp(x + tl.log(tl.abs(y.to(tl.float32))))


def rearrange_chunk(x, chunk_size):
    return rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size)


def rearrange_back(x):
    return rearrange(x, 'b h n c d -> b h (n c) d')


@triton.jit
def chunk_gla_fwd_kernel_h(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    k,  # key [B, H, L, K]
    v,  # value [B, H, L, V]
    g,  # cumulative sum of log decay [B, H, L, K]
    h,  # hidden state [B, H, C, K, V]

    initial_state,  # initial state of the chunk [B, H, K, V]
    final_state,  # final state of the chunk [B, H, K, V]

    s_k_h,  # stride size: L * K
    s_k_t,  # stride size: K
    s_k_d,  # stride size: 1

    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1

    s_hh,
    s_ht,

    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    scale,  # K ** -0.5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (i * BT * K,), (s_k_d,), (i * BT * K - K + i_k * BK,), (BK,), (0,))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        # b_q = tl.load(p_q, boundary_check=(0, 1))

        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

        p_h = tl.make_block_ptr(h + i_bh * s_hh, ((i+1)*K, V), (s_ht, 1), (i*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        # b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
        b_h *= tl.math.exp(b_g)[:, None]
        b_h += tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def chunk_gla_bwd_kernel(
    q,
    g,
    do,  # gradient of output [B, H, L, V]
    dh,

    s_k_h,  # stride size: L * K
    s_k_t,  # stride size: K
    s_k_d,  # stride size: 1

    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1

    s_hh,
    s_ht,

    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    TK,
    scale,  # K ** -0.5
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range((tl.cdiv(T, BT) - 1), -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (i * BT * K,), (s_k_d,), (i * BT * K - K + i_k * BK,), (BK,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, ((i+1)*K, V), (s_ht, 1), (i*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        b_g = tl.math.exp2(tl.load(p_g, boundary_check=(0,)) * inv_ln2)

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        # [K, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [K, V]
        b_dh = b_g[:, None] * b_dh + tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def fwd_decay_cumsum(
    q,
    k,
    g,
    qg,
    kg,
    s_k_h,
    s_k_t,
    s_k_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    cum_decay = tl.zeros([BK], dtype=tl.float32)
    for i in range(i_t * BT, i_t * BT + BT):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))
        p_qg = tl.make_block_ptr(qg + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))

        b_q = tl.load(p_q, boundary_check=(0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        cum_decay += b_g
        b_q *= scale * tl.math.exp(cum_decay)

        tl.store(p_qg, b_q.to(p_qg.dtype.element_ty), boundary_check=(0,))
        tl.store(p_g, cum_decay.to(p_g.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()

    for i in range(i_t * BT, i_t * BT + BT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))
        p_kg = tl.make_block_ptr(kg + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, ((i + 1) * K,), (s_k_d,), (i * K + i_k * BK,), (BK,), (0,))

        b_k = tl.load(p_k, boundary_check=(0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_k *= tl.exp((cum_decay - b_g))
        tl.store(p_kg, b_k.to(p_kg.dtype.element_ty), boundary_check=(0,))


@triton.jit
def bwd_decay_global_cumsum(
    dq_inner,
    dq_inter,
    dk_inner,
    dk_inter,
    q,
    k,
    g,
    dg,
    s_k_h,
    s_k_t,
    s_k_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    last_g = tl.zeros([BK], dtype=tl.float32)
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    for i in range(BT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_dq_inner = tl.make_block_ptr(dq_inner+i_bh*s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_dk_inner = tl.make_block_ptr(dk_inner+i_bh*s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_dq_inter = tl.make_block_ptr(dq_inter+i_bh*s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))
        p_dk_inter = tl.make_block_ptr(dk_inter+i_bh*s_k_h, ((i_t*BT+i+1)*K,), (s_k_d,), ((i_t*BT+i)*K+i_k*BK,), (BK,), (0,))

        b_g = tl.load(p_g, boundary_check=(0,))
        if i == (BT - 1):
            last_g = b_g
        b_dq1 = tl.load(p_dq_inner, boundary_check=(0,))
        b_dq2 = tl.load(p_dq_inter, boundary_check=(0,))
        b_dq2 *= tl.math.exp(b_g)
        b_dq = b_dq1 + b_dq2
        tl.store(p_dq_inter, b_dq.to(p_dq_inter.dtype.element_ty), boundary_check=(0,))
        b_dk1 = tl.load(p_dk_inner, boundary_check=(0,))
        b_dk2 = tl.load(p_dk_inter, boundary_check=(0,))
        b_dk2 *= tl.math.exp(last_g - b_g)
        b_dk = b_dk1 + b_dk2
        tl.store(p_dk_inter, b_dk.to(p_dk_inter.dtype.element_ty), boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0,))
        b_k = tl.load(p_k, boundary_check=(0,))
        b_dg = b_dq * b_q - b_dk * b_k
        cum_grad_dg += b_dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit
def chunk_gla_fwd_kernel_o(
    q,
    q_g,
    k,
    v,
    g,
    h,
    o,
    A,
    s_k_h,  # stride size: L * K
    s_k_t,  # stride size: K
    s_k_d,  # stride size: 1
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    s_A_h,
    s_A_c,
    s_A_t,
    scale,
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q_g = tl.make_block_ptr(q_g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, ((i_t+1)*K, V), (s_h_t, s_h_d), (i_t*K+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BV]
        b_o += tl.dot(b_q_g, b_h, allow_tf32=False)

        # [BT,]
        b_m_gq = tl.max(b_g, 1)
        b_m_gk = tl.min(b_g, 1)
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g - b_m_gq[:, None])).to(b_q.dtype)
        # [BK, BT]
        b_kg = (b_k * tl.trans(tl.exp(b_m_gk[:, None] - b_g))).to(b_k.dtype)
        # [BT,]
        b_sg = tl.dot(b_qg, b_kg, allow_tf32=False)
        b_A += safe_exy(b_m_gq[:, None] - b_m_gk[None, :], b_sg)

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * s_A_h + i_t * s_A_c, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))

    b_A = tl.where(m_s, b_A, 0.) * scale
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bwd_inner_chunk(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    s_k_h,  # stride size: L * K
    s_k_t,  # stride size: K
    s_k_d,  # stride size: 1
    s_A_h,
    s_A_c,
    s_A_t,
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # K ** -0.5
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BTT: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    K: tl.constexpr,  # K
):

    i_c, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    for i in range(i_c * BTT, (i_c + 1) * BTT):
        gq_i = tl.load(g + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + i * K)
        dq_i = tl.zeros([K], dtype=tl.float32)
        for j in range(0, i+1):
            dA_ij = tl.load(dA + i_bh * s_A_h + i_t * s_A_c + i * BT + j)
            k_j = tl.load(k + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + j * K)
            gk_j = tl.load(g + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + j * K)
            dq_i += dA_ij * k_j * tl.math.exp(gq_i - gk_j)
        tl.store(dq + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + i * K, dq_i)

    for j in range(i_c * BTT, (i_c + 1) * BTT):
        gk_j = tl.load(g + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + j * K)
        dk_j = tl.zeros([K], dtype=tl.float32)
        for i in range(j, BT):
            dA_ij = tl.load(dA + i_bh * s_A_h + i_t * s_A_c + i * BT + j)
            q_i = tl.load(q + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + i * K)
            gq_i = tl.load(g + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + i * K)
            dk_j += dA_ij * q_i * tl.math.exp(gq_i - gk_j)
        tl.store(dk + i_bh * s_k_h + i_t * BT * K + tl.arange(0, K) + j * K, dk_j)


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        g = g.to(torch.float32)
        B, H, T, K, V = *q.shape, v.shape[-1]
        ctx.scale = scale

        # inter-chunk
        BT = 64  # chunk_size
        BK, BV = min(K, 64), min(V, 64)
        num_stages = 1
        num_warps = 4

        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)
        grid = (NK, NT, B * H)
        fwd_decay_cumsum[grid](
            q, k, g, q_g, k_g,
            q.stride(1), q.stride(2), q.stride(3),
            B, H, T, scale,
            BT=BT, BK=BK, K=K
        )

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None

        grid = (NV, NK, B * H)
        h = q.new_empty(B, H, NT * K, V)
        chunk_gla_fwd_kernel_h[grid](
            k_g, v, g, h, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            B, H, T, scale,
            BT=BT, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = torch.empty_like(v)
        A = q.new_zeros(B, H, NT, BT, BT)
        grid = (NV, NT, B * H)
        chunk_gla_fwd_kernel_o[grid](
            q, q_g, k, v, g, h, o, A,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            A.stride(1), A.stride(2), A.stride(3),
            scale,
            B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, q_g, k, k_g, v, g, A, initial_state, h)
        return o, final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, q_g, k, k_g, v, g, A, initial_state, h = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale
        BT = 64
        dq = rearrange_back(rearrange_chunk(do, BT) @ rearrange_chunk(h, K).transpose(-1, -2)) * scale

        # inter-chunk
        BK, BV = min(K, 64), min(V, 64)
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 3
        num_warps = 4

        grid = (NV, NK, B * H)
        dh = torch.empty_like(h)

        chunk_gla_bwd_kernel[grid](
            q_g, g, do, dh,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            B, H, T, dh.shape[-2], scale,
            # clamp_min=-3,
            BT=BT, K=K, V=V, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dh = rearrange_chunk(dh, K)
        dk = rearrange_back(torch.einsum('b h n k v, b h n c v -> b h n c k', dh, rearrange_chunk(v, BT)))
        dv = rearrange_back(torch.einsum('b h n k v, b h n c k -> b h n c v', dh, rearrange_chunk(k_g, BT)))

        # intra chunk
        num_chunk = T // BT
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=num_chunk)
        dA2 = (do2 @ v2.transpose(-2, -1)) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dv2 = rearrange(dv2, 'b h n c d -> b h (n c) d', n=num_chunk)

        BK = min(K, 64)
        NK = triton.cdiv(K, BK)
        dk2 = torch.empty_like(k)
        dq2 = torch.empty_like(q)

        BTT = 1
        grid = (BT // BTT, NT, B * H)
        bwd_inner_chunk[grid](
            q, k, g, dA2,
            dq2, dk2,
            q.stride(1), q.stride(2), q.stride(3),
            A.stride(1), A.stride(2), A.stride(3),
            B, H, T, scale,
            BT=BT, BK=BK, BTT=BTT, K=K,
            num_warps=1,
            num_stages=3,
        )

        BK = min(K, 32)
        NK = triton.cdiv(K, BK)
        dg = torch.empty_like(g, dtype=torch.float32)
        grid = (NK, NT, B * H)
        bwd_decay_global_cumsum[grid](
            dq2, dq, dk2, dk, q, k, g, dg,
            q.stride(1), q.stride(2), q.stride(3),
            B, H, T, scale,
            BT=BT, K=K, BK=BK,
            num_warps=1,
            num_stages=1
        )
        dg = rearrange(dg, 'b h (n c) d -> b h n c d', c=BT)

        def rev_cumsum_exclusive(x):
            cumsum_x = x.cumsum(-2)
            rev_cumsum_x = cumsum_x[..., -1, None, :] - cumsum_x
            return rev_cumsum_x

        rev_cumsum_dg = rev_cumsum_exclusive(dg[..., 0, :])
        dg.add_(rev_cumsum_dg.unsqueeze(-2))
        dv.add_(dv2)
        dg = rearrange(dg, 'b h n c d -> b h (n c) d')

        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), None, None, None


def pad(x, chunk_size=64):
    seq_len = x.shape[-2]
    padded_seq_len = ceildiv(seq_len, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - seq_len))
    if x.shape[-1] % 32 != 0:
        x = F.pad(x, (0, 32 - x.shape[-1] % 32))
    return x


def ceildiv(a, b):
    return -(a // -b)


def chunk_gla(
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
    V = v.shape[-1]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = ChunkGLAFunction.apply(
        q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :V]
    if output_final_state:
        return o, final_state
    return o

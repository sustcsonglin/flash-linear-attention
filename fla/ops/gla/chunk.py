# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang
# Gated Linear Attention Transformers with Hardware-Efficient Training: https://arxiv.org/abs/2312.06635
# on-the-fly computation without materializing hidden statets into HBMs

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from fla.ops.utils import contiguous, require_version
from torch.cuda.amp import custom_bwd, custom_fwd

inv_ln2 = 1.44269504

def rearrange_chunk(x, chunk_size):
    return rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size)

def rearrange_back(x):
    return rearrange(x, 'b h n c d -> b h (n c) d')


@triton.jit
def fused_chunk_gla_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    g,  # cumulative sum of log decay [B, H, L, D_head_K]
    h,  # hidden state [B, H, C, DK, DV]

    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1

    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1

    s_hh, 
    s_ht,

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
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV,
                                (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        # b_q = tl.load(p_q, boundary_check=(0, 1))

        d_b = tl.load(p_db).to(tl.float32)

        p_h = tl.make_block_ptr(h + i_bh * s_hh, ((i+1)*DK, DV), (s_ht, 1), (i*DK+i_k*BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        # b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
        b_h *= tl.math.exp(d_b)[:, None]
        b_h += tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False) 

        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_db += BT * DK

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(
            final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty),
                 boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_gla_bwd_kernel(
    q, g,
    do,  # gradient of output [B, H, L, D_head_V]
    dh, 

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1

    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1

    s_hh, s_ht, 

    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    TDK, 
    scale,  # D_head_K ** -0.5
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # [BV, BK]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)
            [:, None] < DK) & (i_v * BV + tl.arange(0, BV)[None, :] < DV)

    p_dh = dh + i_bh * s_hh + (TDK - DK + i_k * BK + tl.arange(0, BK)
                               [:, None]) * DV + i_v * BV + tl.arange(0, BV)[None, :]

    for i in range((tl.cdiv(T, BT) - 1) * BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T),
                                (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV),
                                 (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))

        p_db = g + i_bh * s_qk_h + (i + BT - 1) * \
            s_qk_t + i_k * BK + tl.arange(0, BK)

        d_b = tl.math.exp2(tl.load(p_db) * inv_ln2)

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), mask=mask)
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [DK, DV]
        b_dh = d_b[:, None] * b_dh + tl.dot(b_q, b_do, allow_tf32=False)
        p_dh -= DK * DV
    

@triton.jit
def fwd_decay_cumsum(
    q, k, g,
    qg, kg,
    s_qk_h, s_qk_t, s_qk_d,
    B, H, T, scale,
    BT: tl.constexpr,
    BK: tl.constexpr, DK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(BT):
        _g = tl.load(p_g, mask=mask, other=0)
        _q = tl.load(p_q, mask=mask, other=0)
        cum_decay += _g
        _q *= scale * tl.math.exp(cum_decay)
        tl.store(p_g, cum_decay.to(p_g.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_g += DK
        p_q += DK
        p_qg += DK

    tl.debug_barrier()
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    for i in range(BT):
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0)
        _k *= tl.exp((cum_decay - _g))
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        p_g += DK
        p_k += DK
        p_kg += DK


@triton.jit
def bwd_decay_global_cumsum(
    dq_inner, dq_inter,
    dk_inner, dk_inter,
    q, k, g, dg,
    s_qk_h, s_qk_t, s_qk_d,
    B, H, T, scale,
    BT: tl.constexpr,
    BK: tl.constexpr, DK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT-1, -1, -1):
        _g = tl.load(p_g, mask=mask, other=0)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK


@triton.jit
def fwd_inner_chunk(
    q, k, g, A,
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_A_h,
    s_A_c,
    s_A_t,
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BTT: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    DK: tl.constexpr,  # D_head_K
):

    i_c, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    for i in range(i_c * BTT, (i_c + 1) * BTT):
        q_i = tl.load(q + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK)
        gq_i = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK)
        for j in range(0, i+1):
            k_j = tl.load(k + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK)
            gk_j = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK)
            A_ij = tl.sum(q_i * k_j * tl.math.exp(gq_i - gk_j)) * scale
            tl.store(A + i_bh * s_A_h + i_t * s_A_c + i * BT + j, A_ij)



@triton.jit
def bwd_inner_chunk(
    q, k, g, dA,
    dq, dk, 
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_A_h,
    s_A_c,
    s_A_t,
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    # clamp_min,  # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BTT: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    DK: tl.constexpr,  # D_head_K
):
    
    i_c, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    for i in range(i_c * BTT, (i_c + 1) * BTT):
        gq_i = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK)
        dq_i = tl.zeros([DK], dtype=tl.float32)
        for j in range(0, i+1):
            dA_ij = tl.load(dA + i_bh * s_A_h + i_t * s_A_c + i * BT + j)
            k_j = tl.load(k + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK)
            gk_j = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK)
            dq_i += dA_ij * k_j * tl.math.exp(gq_i - gk_j)
        tl.store(dq + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK, dq_i)

    for j in range(i_c * BTT, (i_c + 1) * BTT):
        gk_j = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK)
        dk_j = tl.zeros([DK], dtype=tl.float32)
        for i in range(j, BT):
            dA_ij = tl.load(dA + i_bh * s_A_h + i_t * s_A_c + i * BT + j)
            q_i = tl.load(q + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK)
            gq_i = tl.load(g + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + i * DK)
            dk_j += dA_ij * q_i * tl.math.exp(gq_i - gk_j)
        tl.store(dk + i_bh * s_qk_h + i_t * BT * DK + tl.arange(0, DK) + j * DK, dk_j)
        
class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        g = g.to(torch.float32)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        ctx.scale = scale

        # inter-chunk
        BT = 64  # chunk_size
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        num_stages = 3
        num_warps = 4

        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)
        grid = (NK, triton.cdiv(seq_len, BT), batch_size * n_heads)
        fwd_decay_cumsum[grid](q, k, g, q_g, k_g, q.stride(1), q.stride(2), q.stride(3),
                               batch_size, n_heads, seq_len, scale, BT=BT, BK=BK, DK=d_head_qk)
        
        if output_final_state:
            final_state = q.new_empty(
                batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None
            
        grid = (NV, NK, batch_size * n_heads)
        h = q.new_empty(batch_size, n_heads, triton.cdiv(seq_len, BT) * d_head_qk, d_head_v)
        fused_chunk_gla_fwd_kernel[grid](
            k_g, v, g, h, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = rearrange(q_g, 'b h (n c) d -> b h n c d', c = BT) @ rearrange(h, 'b h (n c) d -> b h n c d', c = d_head_qk)
        o = rearrange(o, 'b h n c d -> b h (n c) d')  

        chunk_size = 64
        num_chunk = seq_len // chunk_size
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        BK = min(d_head_qk, 64)
        NK = triton.cdiv(d_head_qk, BK)
        A = q.new_zeros(batch_size, n_heads,
                        triton.cdiv(seq_len, BT), BT, BT)
        BTT = 1
        grid = (chunk_size // BTT, triton.cdiv(seq_len, BT), batch_size * n_heads)
        fwd_inner_chunk[grid](q, k, g, A,
                              q.stride(1), q.stride(2), q.stride(3),
                              A.stride(1), A.stride(2), A.stride(3),
                              batch_size, n_heads, seq_len, scale,  BT=BT, BK=BK, BTT=BTT, DK=d_head_qk, num_stages=3, num_warps=1)        

        o2 = A @ v2    
        o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
        o2 += o
        ctx.save_for_backward(q, q_g, k, k_g, v, g, A, initial_state, h)
        return o2.to(v), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, q_g, k, k_g, v, g, A, initial_state, h = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale
        BT = 64
        dq = rearrange_back(rearrange_chunk(do, BT) @ rearrange_chunk(h, d_head_qk).transpose(-1, -2)) * scale
       
        # inter-chunk
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 3
        num_warps = 4

        grid = (NV, NK, batch_size * n_heads)
        dh = torch.empty_like(h)

        fused_chunk_gla_bwd_kernel[grid](
            q_g, g, do, dh, 
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            batch_size, n_heads, seq_len, dh.shape[-2], scale,
            # clamp_min=-3,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dh = rearrange_chunk(dh, d_head_qk)
        dk = rearrange_back(torch.einsum('b h n k v, b h n c v -> b h n c k', dh, rearrange_chunk(v, BT)))
        dv = rearrange_back(torch.einsum('b h n k v, b h n c k -> b h n c v', dh, rearrange_chunk(k_g, BT)))

        # intra chunk
        num_chunk = seq_len // BT
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=num_chunk)
        dA2 = (do2 @ v2.transpose(-2, -1)) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dv2 = rearrange(dv2, 'b h n c d -> b h (n c) d', n=num_chunk)

        BK = min(d_head_qk, 64)
        NK = triton.cdiv(d_head_qk, BK)
        dk2 = torch.empty_like(k)
        dq2 = torch.empty_like(q)

        BTT = 1
        grid = (BT // BTT, triton.cdiv(seq_len, BT), batch_size * n_heads)
        bwd_inner_chunk[grid](q, k, g, dA2,
                              dq2, dk2,
                              q.stride(1), q.stride(2), q.stride(3),
                              A.stride(1), A.stride(2), A.stride(3),
                              batch_size, n_heads, seq_len, scale,  BT=BT, BK=BK, BTT=BTT, DK=d_head_qk, num_stages=3,
                              num_warps=1)

        BK = min(d_head_qk, 32)
        NK = triton.cdiv(d_head_qk, BK)
        dg = torch.empty_like(g, dtype=torch.float32)
        grid = (NK, triton.cdiv(seq_len, BT), batch_size * n_heads)
        bwd_decay_global_cumsum[grid](dq2, dq, dk2, dk, q, k, g, dg,
                                      q.stride(1), q.stride(2), q.stride(3),
                                      batch_size, n_heads, seq_len, scale,
                                      BT=BT, DK=d_head_qk, BK=BK,
                                      num_warps=1,
                                      num_stages=1)
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
    d_head_v = v.shape[-1]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = ChunkGLAFunction.apply(
        q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :d_head_v]
    if output_final_state:
        return o, final_state
    return o


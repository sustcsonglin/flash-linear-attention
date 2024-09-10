# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV", "USE_G", 'USE_GK', 'USE_GV'],
)
@triton.jit
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    gk,
    gv,
    h0,
    ht,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + i_bh * T + last_idx)
            b_h *= tl.exp(b_g_last)
            p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK: 
            p_gk_last = tl.make_block_ptr(gk + i_bh * s_qk_h, (T * K,), (s_qk_d,), (last_idx * K + i_k * BK,), (BK,), (0,))
            b_gk_last = tl.load(p_gk_last, boundary_check=(0,))
            b_h *= tl.exp(b_gk_last)[:, None]
            
            p_gk = tl.make_block_ptr(gk + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv_last = tl.make_block_ptr(gv + i_bh * s_vo_h, (T * V,), (s_vo_d,), (last_idx * V + i_v * BV,), (BV,), (0,))
            b_gv_last = tl.load(p_gv, boundary_check=(0,))
            b_h *= tl.exp(b_gv_last)[None, :]
    
            p_gv = tl.make_block_ptr(gv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)            
            
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV",  "USE_G", 'USE_GK', 'USE_GV'],
)
@triton.jit
def chunk_bwd_kernel_dh(
    q,
    g,
    gk, 
    gv,
    do,
    dh,
    dht,
    dh0,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    LOAD_FINAL_STATE_GRADIENT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if LOAD_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
    
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]        
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_q = (b_q * tl.exp(b_g)[None, :]).to(b_q.dtype)
            b_g_last = tl.load(g + i_bh * T + last_idx)
            b_dh *= tl.exp(b_g_last)
        
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = (b_q * tl.exp(b_gk)).to(b_q.dtype)
            p_gk_last = tl.make_block_ptr(gk + i_bh * s_qk_h, (T * K,), (s_qk_d,), (last_idx * K + i_k * BK,), (BK,), (0,))
            b_gk_last = tl.load(p_gk_last, boundary_check=(0,))
            b_dh *= tl.exp(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = (b_do * tl.exp(b_gv)).to(b_do.dtype)
            p_gv_last = tl.make_block_ptr(gv + i_bh * s_vo_h, (T * V,), (s_vo_d,), (last_idx * V + i_v * BV,), (BV,), (0,))
            b_gv_last = tl.load(p_gv, boundary_check=(0,))
            b_dh *= tl.exp(b_gv_last)[None, :]
        
        b_dh += tl.dot(b_q, b_do, allow_tf32=False) 
    
    
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))

    


def chunk_fwd_h_fn(k, v, g, gk, gv, BT, h0, output_final_state):
    B, H, T, K, V = *k.shape, v.shape[-1]
    ht = None
    if output_final_state:
        ht = k.new_empty(B, H, K, V, dtype=torch.float32)

    BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    h = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)

    USE_G, USE_GK, USE_GV = g is not None, gk is not None, gv is not None

    chunk_fwd_kernel_h[grid](
        k, v, h, g, gk, gv, h0, ht,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=output_final_state,
        USE_G=USE_G, USE_GK=USE_GK, USE_GV=USE_GV
    )
    return h, ht



def chunk_bwd_dh_fn(q, k, v, g, gk, gv, do, h0, dht, BT, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    dh = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    if h0 is not None:
        dh0 = torch.empty_like(h0, dtype=torch.float32)
    else:
        dh0 = None
    USE_GATE = (g is not None) or (gk is not None) or (gv is not None)
    assert not (USE_GATE and dht is not None), "Cannot load final state gradient and use gates at the same time"
    chunk_bwd_kernel_dh[grid](
        q, g, gk, gv, do, dh, dht, dh0,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None,
        STORE_INITIAL_STATE_GRADIENT=dh0 is not None,
        LOAD_FINAL_STATE_GRADIENT=dht is not None
    )
    return dh, dh0




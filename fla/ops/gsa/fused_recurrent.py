# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.jit
def fused_recurrent_gsa_inference_kernel(
    q,
    k,
    v,
    s,
    g,
    o,
    hk0,
    hv0,
    hkt,
    hvt,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    b_s = tl.load(s + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.load(g + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.exp(b_g)

    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + tl.arange(0, M)[:, None]
        # [BK,]
        mask_k = o_k < K
        # [M, BK]
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        # [M, BK]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.).to(tl.float32)
        # [BK,]
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
        # [BV,]
        mask_v = o_v < V
        # [BV, M]
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        # [BV, M]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        # [BV,]
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)


@triton.jit
def fused_recurrent_gsa_fwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
        p_o = o + (i_bh + i_k * B*H) * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)

        if USE_GK:
            p_gk = gk + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        if USE_GV:
            p_gv = gv + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    else:
        p_q = q + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k*B + i_b) * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        if USE_GK:
            p_gk = gk + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_b * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk)[None, :]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv)[:, None]
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit
def fused_recurrent_gsa_bwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    h0,
    do,
    dq,
    dk,
    dv,
    dh0,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
        p_dq = dq + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        if USE_GK:
            p_gk = gk + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
        if USE_GV:
            p_gv = gv + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    else:
        p_q = q + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_b * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v*B + i_b) * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GK:
            p_gk = gk + i_b * T*H*K + ((T-1) * H*K if REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_b * T*H*V + ((T-1) * H*V if REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv)[None, :]
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V
        p_do += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V
        p_dq += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H)*V

    # sync threads
    tl.debug_barrier()

    if HEAD_FIRST:
        p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
        p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
        p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
        p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
        p_dk = dk + (i_v * B*H + i_bh) * T*K + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
        p_dv = dv + (i_k * B*H + i_bh) * T*V + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
        if USE_GK:
            p_gk = gk + i_bh * T*K + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
        if USE_GV:
            p_gv = gv + i_bh * T*V + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    else:
        p_q = q + i_b * T*H*K + ((T - 1) if not REVERSE else 0) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_b * T*H*K + ((T - 1) * H*K if not REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_b * T*H*V + ((T - 1) * H*V if not REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + i_b * T*H*V + ((T - 1) * H*V if not REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dk = dk + (i_v*B + i_b) * T*H*K + ((T - 1) * H*K if not REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k*B + i_b) * T*H*V + ((T - 1) * H*V if not REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)
        if USE_GK:
            p_gk = gk + i_b * T*H*K + ((T - 1) * H*K if not REVERSE else 0) + i_h * K + i_k * BK + tl.arange(0, BK)
        if USE_GV:
            p_gv = gv + i_b * T*H*V + ((T - 1) * H*V if not REVERSE else 0) + i_h * V + i_v * BV + tl.arange(0, BV)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gv)[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*K
        p_k += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*K
        p_v += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*V
        p_do += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*V
        p_dk += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*K
        p_dv += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*V
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*K
        if USE_GV:
            p_gv += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H)*V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gsa_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    head_first: bool = True
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[1] if head_first else q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NG = HQ // H

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)

    o = v.new_empty(B, HQ, T, V) if head_first else v.new_empty(B, T, HQ, V)
    grid = (B * HQ,)
    fused_recurrent_gsa_inference_kernel[grid](
        q,
        k,
        v,
        s,
        g,
        o,
        hk0,
        hv0,
        hkt,
        hvt,
        scale=scale,
        K=K,
        V=V,
        M=M,
        BK=BK,
        BV=BV,
        NG=NG
    )
    return o, (hkt, hvt)


def fused_recurrent_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    reverse: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]

    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)

    if head_first:
        ok = q.new_empty(NK, B, H, T, M, dtype=torch.float)
    else:
        ok = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, B*H)
    fused_recurrent_gsa_fwd_kernel[grid](
        q,
        k,
        s,
        gk,
        gv,
        ok,
        hk0,
        hkt,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_INITIAL_STATE=hk0 is not None,
        STORE_FINAL_STATE=hkt is not None,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    ok = ok.sum(0)

    qv = ok.softmax(-1, dtype=torch.float)
    if head_first:
        ov = q.new_empty(NM, B, H, T, V, dtype=torch.float)
    else:
        ov = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    gk, gv = g, None
    grid = (NV, NM, B*H)
    fused_recurrent_gsa_fwd_kernel[grid](
        qv,
        s,
        v,
        gk,
        gv,
        ov,
        hv0,
        hvt,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_INITIAL_STATE=hv0 is not None,
        STORE_FINAL_STATE=hvt is not None,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: Optional[torch.Tensor] = None,
    hv0: Optional[torch.Tensor] = None,
    ok: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    scale: float = 1.,
    reverse: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor]:
    if head_first:
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    BK, BV, BM = min(K, 64), min(V, 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    if head_first:
        dqv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dv = q.new_empty(NM, B, H, T, V, dtype=torch.float)
    else:
        dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dhk0 = torch.empty_like(hk0)if hk0 is not None else None
    dhv0 = torch.empty_like(hv0)if hv0 is not None else None

    gk, gv = g, None
    grid = (NV, NM, B*H)
    fused_recurrent_gsa_bwd_kernel[grid](
        qv,
        s,
        v,
        gk,
        gv,
        hv0,
        do,
        dqv,
        dsv,
        dv,
        dhv0,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_INITIAL_STATE=hv0 is not None,
        REVERSE=reverse,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgk = chunk_global_cumsum(dqv * qv.float() - dsv * s.float(), reverse=True, head_first=head_first)

    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    if head_first:
        dq = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dk = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, H, T, M, dtype=torch.float)
    else:
        dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, B*H)
    fused_recurrent_gsa_bwd_kernel[grid](
        q,
        k,
        s,
        gk,
        gv,
        hk0,
        dok,
        dq,
        dk,
        dsk,
        dhk0,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_INITIAL_STATE=hk0 is not None,
        REVERSE=reverse,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)

    dgv = chunk_global_cumsum(dok.float() * ok.float() - dsk * s.float(), reverse=True, head_first=head_first)

    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)

    return dq, dk, dv, ds, dg, dhk0, dhv0


class FusedRecurrentGSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[float] = None,
        hk0: Optional[torch.Tensor] = None,
        hv0: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        head_first: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        T = q.shape[2] if head_first else q.shape[1]
        if T == 1 and not q.requires_grad:
            o, (hkt, hvt) = fused_recurrent_gsa_inference(
                q=q,
                k=k,
                v=v,
                s=s,
                g=g,
                initial_state=(hk0, hv0),
                output_final_state=output_final_state,
                scale=scale,
                head_first=head_first
            )
            return o, (hkt, hvt)
        ok, hkt, qv, ov, hvt = fused_recurrent_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            reverse=reverse,
            head_first=head_first
        )
        ctx.save_for_backward(q, k, v, s, g, qv, hk0, hv0, ok)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.head_first = head_first
        return ov.to(q.dtype), (hkt, hvt)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        head_first = ctx.head_first
        dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            qv=qv,
            hk0=hk0,
            hv0=hv0,
            ok=ok,
            do=do,
            scale=scale,
            reverse=reverse,
            head_first=head_first
        )
        return dq.to(q), dk.to(k), dv.to(v), ds.to(s), dg.to(g), None, dhk0, dhv0, None, None, None


def fused_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    reverse: Optional[bool] = False,
    head_first: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`.
            Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[B, H, K, M]` and `[B, H, M, V]`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = (None, None)
    o, final_state = FusedRecurrentGSAFunction.apply(
        q, k, v, s, g, scale, *initial_state, output_final_state, reverse, head_first
    )
    return o, final_state

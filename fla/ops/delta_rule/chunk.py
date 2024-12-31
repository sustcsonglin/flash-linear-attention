# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.delta_rule.wy_fast import (bwd_prepare_wy_repr,
                                        fwd_prepare_wy_repr, fwd_recompute_w_u)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dv, chunk_bwd_dqkwg


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    v_new,
    h,
    h0,
    ht,
    offsets,
    chunk_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_nh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+i_nh*T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT+i_c*BC, i_v * BV), (BC, BV), (1, 0))
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BC, BK]
            b_d = tl.load(p_d, boundary_check=(0, 1))
            # [BC, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype))
            # [BK, BV]
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    offsets,
    chunk_offsets,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(dh + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(q + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k + i_nh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2 + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            # [BT, V]
            b_do = tl.load(p_do, boundary_check=(0, 1))

            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)

            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d, b_dv.to(b_q.dtype), allow_tf32=False)
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)
    chunk_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        d=w,
        v_new=v_new,
        h=h,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    return h, v_new, final_state



def chunk_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, do.shape[-1]
    else:
        B, T, H, K, V = *q.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT = B, triton.cdiv(T, BT)
    else:
        N, NT = len(offsets) - 1, chunk_offsets[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    # H100
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        dh = q.new_empty(B, H, NT, K, V)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    grid = (NK, NV, N * H)
    chunk_delta_rule_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        d=w,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dh, dh0, dv2


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # obtain WY representation. u is actually the new v.
    w, u, A = fwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )

    h, v_new, final_state = chunk_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return o, A, final_state


def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(
        k=k,
        v=v,
        beta=beta,
        A=A,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    h, v_new, _ = chunk_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=False,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    dv = chunk_bwd_dv(
        q=q,
        k=k,
        do=do,
        g=None,
        dh=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dh, dh0, dv = chunk_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    dq, dk, dw, _ = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        h=h,
        w=w,
        dv=dv,
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk2, dv, db = bwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=dv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(triton.next_power_of_2(T), 16))

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        o, A, final_state = chunk_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, beta, A, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, beta, A, initial_state = ctx.saved_tensors
        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=ctx.offsets,
            indices=ctx.indices,
            head_first=ctx.head_first,
            chunk_size=ctx.chunk_size
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None, None


def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        offsets (Optional[torch.LongTensor]):
            Offsets of shape `[N+1]` defining the bos/eos positions of `N` variable-length sequences in the batch.
            For example,
            if `offsets` is `[0, 1, 3, 6, 10, 15]`, there are `N=5` sequences with lengths 1, 2, 3, 4 and 5 respectively.
            If provided, the inputs are concatenated and the batch size `B` is expected to be 1.
            Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import chunk_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_delta_rule(q, k, v, beta,
                                     initial_state=h0,
                                     output_final_state=True,
                                     head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `offsets` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, offsets with 5 start/end positions are expected
        >>> offsets = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_delta_rule(q, k, v, beta,
                                             initial_state=h0,
                                             output_final_state=True,
                                             offsets=offsets,
                                             head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if offsets is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `offsets`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(offsets) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(offsets) - 1} rather than {initial_state.shape[0]}.")
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        offsets,
        head_first
    )
    return o, final_state

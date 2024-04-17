# -*- coding: utf-8 -*-

# Copyright (c) 2024, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.utils import chunk_reversed_cumsum_fwd
from fla.utils import contiguous

# on-the-fly computation without materializing hidden statets into HBMs


@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    w,  # log gate [B, H, L, D_head_K]
    u,  # bonus [B, H, D_head_K]
    o,  # output [B, H, L, D_head_V]
    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,
    final_state,  # final hidden state [B, H, D_head_K, D_head_V]

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
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)

    p_w = w + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)

    p_u = u + i_h * DK + tl.arange(0, BK) + i_k * BK

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV

    h = tl.zeros([BV, BK], dtype=tl.float32)

    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    _u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        _w = tl.exp(_w)
        _kv = _k[None, :] * _v[:, None]
        _o = (h + _kv * _u[None, :]) * _q[None, :]
        _o = tl.sum(_o, axis=1)
        h = h * _w[None, :]
        h += _kv
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -DK if REVERSE else DK
        p_k += -DK if REVERSE else DK
        p_o += -DV if REVERSE else DV
        p_v += -DV if REVERSE else DV
        p_w += -DK if REVERSE else DK

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dq(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    w,  # log gate [B, H, L, D_head_K]
    u,  # bonus [B, H, D_head_K]

    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dq_aux,  # gradient of query_aux [NV, B, H, L, D_head_K]

    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,

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
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_dq_aux = dq_aux + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_w = w + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_u = u + i_h * DK + tl.arange(0, BK) + i_k * BK

    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    _u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _kv = _k[None, :] * _v[:, None]
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        _w = tl.exp(_w)
        h_q = h * _do[:, None]
        _dq = tl.sum(h_q + _kv * _u[None, :] * _do[:, None], axis=0)
        _dq *= scale
        _dq_aux = tl.sum(h_q, axis=0)
        h = h * _w[None, :]
        h += _kv
        tl.store(p_dq, _dq.to(p_dq.dtype.element_ty), mask=mask_bk)
        tl.store(p_dq_aux, _dq_aux.to(p_dq_aux.dtype.element_ty), mask=mask_bk)
        p_k += -DK if REVERSE else DK
        p_do += -DV if REVERSE else DV
        p_v += -DV if REVERSE else DV
        p_w += -DK if REVERSE else DK
        p_dq += -DK if REVERSE else DK
        p_dq_aux += -DK if REVERSE else DK


@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    w,  # log gate [B, H, L, D_head_K]
    u,  # bonus [B, H, D_head_K]

    do,  # gradient of output [B, H, L, D_head_V]
    dk,
    dk_aux,
    dv,

    # initial hidden state initialization [B, H, D_head_K, D_head_V]
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
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_dk_aux = dk_aux + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_w = w + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    p_u = u + i_h * DK + tl.arange(0, BK) + i_k * BK
    _u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    for i in range(T-1, -1, -1):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _dkv = _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        tl.store(p_dk_aux, d_k.to(p_dk_aux.dtype.element_ty), mask=mask_bk)
        d_k += tl.sum(_dkv * _u[:, None] * _v[None, :], axis=1)
        d_v = tl.sum((d_h + (_dkv * _u[:, None])) * _k[:, None], axis=0)

        _w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        _w = tl.exp(_w)
        d_h *= _w[:, None]
        d_h += _dkv

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do += DV if REVERSE else -DV
        p_q += DK if REVERSE else -DK
        p_k += DK if REVERSE else -DK
        p_v += DV if REVERSE else -DV
        p_dk += DK if REVERSE else -DK
        p_dk_aux += DK if REVERSE else -DK
        p_dv += DV if REVERSE else -DV
        p_w += DK if REVERSE else -DK


class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, r, k, v, w, u, scale=None, initial_state=None, output_final_state=False, reverse=False):
        # alias
        q = r
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        # default scale
        if scale is None:
            scale = d_head_qk ** -0.5

        BK, BV = min(triton.next_power_of_2(d_head_qk), 32), min(triton.next_power_of_2(d_head_v), 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len,
                        d_head_v, dtype=torch.float32)

        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        grid = (NV, NK, batch_size * n_heads)
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, w, u, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        # we do not need the gradient of the final state from the next chunk
        # similiar to Trunctated BPTT
        if final_state is not None:
            final_state = final_state.detach()
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, w, u, initial_state, o = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(triton.next_power_of_2(d_head_qk), 16), min(triton.next_power_of_2(d_head_v), 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1
        dq = q.new_empty(NV, batch_size, n_heads,  seq_len,
                         d_head_qk, dtype=torch.float32)
        dq_aux = torch.empty_like(dq)
        grid = (NV, NK, batch_size * n_heads)

        fused_recurrent_rwkv6_bwd_kernel_dq[grid](
            k, v, w, u, do, dq, dq_aux, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse,
        )
        dq = dq.sum(0).to(q)
        dq_aux = dq_aux.sum(0)

        BK, BV = min(triton.next_power_of_2(d_head_qk), 32), min(triton.next_power_of_2(d_head_v), 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len,
                         d_head_qk, dtype=torch.float32)
        dk_aux = q.new_empty(NV, batch_size, n_heads,  seq_len,
                             d_head_qk, dtype=torch.float32)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len,
                         d_head_v, dtype=torch.float32)
        grid = (NV, NK, batch_size * n_heads)

        fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
            q, k, v, w, u, do, dk, dk_aux, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse,
        )
        dk = dk.sum(0).to(k)
        dv = dv.sum(0).to(v)
        dk_aux = dk_aux.sum(0)

        dw = (dq_aux * q * scale)[:, :, 1:] - (dk_aux * k)[:, :, 0:-1]
        dw = torch.nn.functional.pad(dw, (0, 0, 0, 1, 0, 0, 0, 0), value=0)
        dw = chunk_reversed_cumsum_fwd(dw).to(w)

        du = ((do * v).sum(-1)[..., None] * k * q * scale).sum([0, -2]).to(u)
        return dq, dk, dv, dw, du, None, None, None, None


# if scale is None, use d_head_qk ** -0.5 by default. Otherwise specify the scale yourself. e.g. scale = 1.0
def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)`
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    if scale == -1:
        scale = r.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state)
    return o, final_state

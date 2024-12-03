# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    # decay rate given the head index
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        p_o = o + (i_k * B*H + i_nh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_h = b_b * b_h + b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q,
    k,
    v,
    h0,
    do,
    dq,
    dk,
    dv,
    dh0,
    dht,
    offsets,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    if HEAD_FIRST:
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_do = do + i_nh * T*V + ((T-1) * V if REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_dq = dq + (i_v * B*H + i_nh) * T*K + ((T-1) * K if REVERSE else 0) + i_k * BK + tl.arange(0, BK)
    else:
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_h = b_b * b_h + b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_do += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_dq += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K

    # sync threads
    tl.debug_barrier()

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_do = do + i_nh * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
        p_dk = dk + (i_v * B*H + i_nh) * T*K + ((T - 1) * K if not REVERSE else 0) + i_k * BK + tl.arange(0, BK)
        p_dv = dv + (i_k * B*H + i_nh) * T*V + ((T - 1) * V if not REVERSE else 0) + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos + ((T - 1) if not REVERSE else 0))*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_do = do + (bos + ((T - 1) if not REVERSE else 0))*H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dk = dk + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_dv = dv + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_h, other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_dh *= b_b
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_k += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_v += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V
        p_do += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V
        p_dk += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * K
        p_dv += (1 if REVERSE else -1) * (1 if HEAD_FIRST else H) * V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def fused_recurrent_retention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)

    grid = (NV, NK, N * H)
    fused_recurrent_retention_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        h0,
        ht,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    o = o.sum(0)
    return o.to(v.dtype), ht


def fused_recurrent_retention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1

    BK, BV = min(K, 64), min(V, 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)
    h0 = initial_state
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None

    grid = (NV, NK, N * H)
    fused_recurrent_retention_bwd_kernel[grid](
        q,
        k,
        v,
        h0,
        do,
        dq,
        dk,
        dv,
        dh0,
        dht,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    return dq, dk, dv, dh0


class FusedRecurrentRetentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        o, ht = fused_recurrent_retention_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.offsets = offsets
        ctx.head_first = head_first
        return o.to(v.dtype), ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, initial_state = ctx.saved_tensors
        dq, dk, dv, dh0 = fused_recurrent_retention_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            offsets=ctx.offsets,
            head_first=ctx.head_first
        )
        return dq.to(q), dk.to(k), dv.to(v), None, dh0, None, None, None, None


def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
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
        >>> from fla.ops.retention import fused_recurrent_retention
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_retention(q, k, v,
                                              initial_state=h0,
                                              output_final_state=True,
                                              head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `offsets` is required
        >>> q, k, v = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v))
        # for a batch with 4 sequences, offsets with 5 start/end positions are expected
        >>> offsets = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_retention(q, k, v,
                                                      initial_state=h0,
                                                      output_final_state=True,
                                                      offsets=offsets,
                                                      head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if offsets is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `offsets`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(offsets) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(offsets) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRetentionFunction.apply(
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        reverse,
        offsets,
        head_first
    )
    return o, final_state

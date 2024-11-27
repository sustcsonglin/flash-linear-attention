# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=['BT']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=['BT']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_reversed_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_z = tl.sum(b_s, axis=0)
    b_o = b_z[None] - tl.cumsum(b_s, axis=0) + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S', 'BT']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S', 'BT']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_reversed_cumsum_vector_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_z += tl.sum(b_s, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + start*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s,
    z,
    offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b).to(tl.int32), tl.load(offsets + i_b + 1).to(tl.int32)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start

    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + start*H*S + i_h * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    if offsets is not None:
        B = len(offsets) - 1
    BT = chunk_size
    if offsets is not None:
        B = len(offsets) - 1
        NT = sum(triton.cdiv(i, BT) for i in (offsets[1:] - offsets[:-1]).tolist())
    else:
        NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    grid = (NT, B * H)
    if reverse:
        chunk_local_reversed_cumsum_scalar_kernel[grid](
            g_org,
            g,
            offsets,
            T=T,
            H=H,
            BT=BT,
            HEAD_FIRST=head_first
        )
    else:
        chunk_local_cumsum_scalar_kernel[grid](
            g_org,
            g,
            offsets,
            T=T,
            H=H,
            BT=BT,
            HEAD_FIRST=head_first
        )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    if offsets is not None:
        B = len(offsets) - 1
        NT = sum(triton.cdiv(i, BT) for i in (offsets[1:] - offsets[:-1]).tolist())
    else:
        NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    if reverse:
        chunk_local_reversed_cumsum_vector_kernel[grid](
            g_org,
            g,
            offsets,
            T=T,
            H=H,
            S=S,
            BT=BT,
            HEAD_FIRST=head_first
        )
    else:
        chunk_local_cumsum_vector_kernel[grid](
            g_org,
            g,
            offsets,
            T=T,
            H=H,
            S=S,
            BT=BT,
            HEAD_FIRST=head_first
        )
    return g


@contiguous
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    if offsets is not None:
        assert not head_first, "Sequences with variable lengths are not supported for head-first mode"
        assert g.shape[0] == 1, "Only batch size 1 is supported when offsets are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g, chunk_size, reverse, offsets, head_first)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g, chunk_size, reverse, offsets, head_first)
    else:
        raise ValueError(f"Unsupported input shape {g.shape}. "
                         f"which should be (B, H, T, dim) if `head_first=True` "
                         f"or (batch_size, num_heads, seq_len) otherwise")


@contiguous
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    if offsets is not None:
        B = len(offsets) - 1
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    if reverse:
        chunk_global_reversed_cumsum_scalar_kernel[grid](
            s,
            z,
            offsets,
            T=T,
            H=H,
            HEAD_FIRST=head_first
        )
    else:
        chunk_global_cumsum_scalar_kernel[grid](
            s,
            z,
            offsets,
            T=T,
            H=H,
            HEAD_FIRST=head_first
        )
    return z


@contiguous
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    BS = min(32, S)
    if offsets is not None:
        B = len(offsets) - 1
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    if reverse:
        chunk_global_reversed_cumsum_vector_kernel[grid](
            s,
            z,
            offsets,
            T=T,
            H=H,
            S=S,
            BS=BS,
            HEAD_FIRST=head_first
        )
    else:
        chunk_global_cumsum_vector_kernel[grid](
            s,
            z,
            offsets,
            T=T,
            H=H,
            S=S,
            BS=BS,
            HEAD_FIRST=head_first
        )
    return z


@contiguous
def chunk_global_cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    reverse: bool = False,
    offsets: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    if offsets is not None:
        assert not head_first, "Sequences with variable lengths are not supported for head-first mode"
        assert s.shape[0] == 1, "Only batch size 1 is supported when offsets are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(s, dtype, reverse, offsets, head_first)
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(s, dtype, reverse, offsets, head_first)
    else:
        raise ValueError(f"Unsupported input shape {s.shape}. "
                         f"which should be [B, H, T]/[B, H, T, D] if `head_first=True` "
                         f"or [B, T, H]/[B, T, H, D] otherwise")

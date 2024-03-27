# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import contiguous


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
@triton.jit
def logcumsumexp_fwd_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_mp = tl.full([S,], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S,], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

        # [BT, S]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [S,]
        b_mc = tl.max(b_s, 0)
        # workaround for compiler bugs
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        # [BT, S]
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        # [S,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def softmax_fwd_kernel(
    s,
    p,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))

    # [BT, S]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    # [BT]
    b_m = tl.max(b_s, 1)

    # [BT, BS]
    b_s = tl.exp(b_s - b_m[:, None])
    b_z = tl.sum(b_s, 1)
    b_p = tl.where(b_s != 0, b_s / b_z[:, None], 0.)
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    # [BT, BS]
    b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
    b_dp = tl.load(p_dp, boundary_check=(0, 1)).to(tl.float32)
    # [BT,]
    b_pp = tl.sum(b_p * b_dp, 1)
    # [BT, BS]
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))


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
@triton.jit
def cumsum_fwd_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
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
@triton.jit
def cumsum_bwd_kernel(
    ds,
    dz,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_ds = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_dz = tl.make_block_ptr(dz + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_dz = tl.load(p_dz, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_ds[None, :] + tl.dot(m_s, b_dz, allow_tf32=False)
        tl.store(p_ds, b_c.to(p_ds.dtype.element_ty), boundary_check=(0, 1))

        if i_t >= 0:
            b_ds += tl.sum(b_dz, 0)


@contiguous
def cumsum_fwd(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32

    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    cumsum_fwd_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z


@contiguous
def cumsum_bwd(
    dz: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = dz.shape
    BS = 32

    dtype = dtype or dz.dtype
    grid = (triton.cdiv(S, BS), B * H)
    ds = torch.empty_like(dz, dtype=dtype)
    cumsum_bwd_kernel[grid](
        ds, dz,
        ds.stride(1), ds.stride(2), ds.stride(3),
        T=T, S=S, BS=BS
    )
    return ds


class CumsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, dtype):
        z = cumsum_fwd(s, dtype)
        ctx.dtype = dtype
        return z

    @staticmethod
    def backward(ctx, dz):
        ds = cumsum_bwd(dz, ctx.dtype)
        return ds, None


def cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return CumsumFunction.apply(s, dtype)

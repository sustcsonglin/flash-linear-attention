# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.utils import contiguous


# Inspired by "THE WY REPRESENTATION FOR PRODUCTS OF HOUSEHOLDER MATRICES" https://epubs.siam.org/doi/pdf/10.1137/0908009
# o: cumprod
# o2: cumprodsum
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def fwd_prepare_wy_repr_kernel(
    k, v, beta, o, o2,
    NT, DK, DV, T, 
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = k + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :]
    p_v = v + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]
    p_beta = beta + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask_bt = (tl.arange(0, BT) + i_t * BT) < T
    mask_bk = tl.arange(0, BK) < DK
    mask_bv = tl.arange(0, BV) < DV
    mask_bk = mask_bk[None, :] & mask_bt[:, None]
    mask_bv = mask_bv[None, :] & mask_bt[:, None]
    b_k = tl.load(p_k, mask=mask_bk)
    b_v = tl.load(p_v, mask=mask_bv)
    b_beta = tl.load(p_beta, mask=mask_bt).to(tl.float32)
    b_v = (b_v * b_beta[:, None])
    A = tl.dot(b_k, tl.trans(b_k), allow_tf32=False) * b_beta[:, None]
    A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], A, 0)
    b_o = b_k.to(tl.float32) * b_beta[:, None]

    for i in range(BT):
        mask = tl.arange(0, BT) == i
        attn = tl.sum(tl.where(mask[:, None], A, 0), axis=0)
        new_o = tl.sum(attn[:, None] * b_o, axis=0)
        new_v = tl.sum(attn[:, None] * b_v, axis=0)
        b_o = tl.where(mask[:, None], b_o - new_o[None, :], b_o)
        b_v = tl.where(mask[:, None], b_v - new_v[None, :], b_v)

    p_o = o + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:,  None]) * DK + tl.arange(0, BK)[None, :]
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bk)
    p_o2 = o2 + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]
    tl.store(p_o2, b_v.to(p_o2.dtype.element_ty), mask=mask_bv)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def bwd_prepare_wy_repr_kernel(
    k, v, beta,
    o, o2, do, do2,
    dk, dv, dbeta,
    NT, DK, DV, T,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = k + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :]
    p_do = do + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :]
    p_do2 = do2 + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]

    p_beta = beta + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask_bt = (tl.arange(0, BT) + i_t * BT) < T
    mask_bk = (tl.arange(0, BK) < DK)[None, :] & mask_bt[:, None]
    mask_bv = (tl.arange(0, BV) < DV)[None, :] & mask_bt[:, None]
    b_k, b_beta = tl.load(p_k, mask=mask_bk), tl.load(p_beta, mask=mask_bt)

    b_beta = b_beta.to(tl.float32)
    A = tl.dot(b_k, tl.trans(b_k), allow_tf32=False) * b_beta[:, None]
    A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], A, 0)
    b_do = tl.load(p_do, mask=mask_bk).to(tl.float32)
    b_dv = tl.load(p_do2, mask=mask_bv).to(tl.float32)
    dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i in range(BT-1, -1, -1):
        mask = tl.arange(0, BT) == i
        attn = tl.sum(tl.where(mask[:, None], A, 0), axis=0)
        do_ = tl.sum(tl.where(mask[:, None], b_do, 0), axis=0)
        dv_ = tl.sum(tl.where(mask[:, None], b_dv, 0), axis=0)
        b_do = b_do - attn[:, None] * do_[None, :]
        b_dv = b_dv - attn[:, None] * dv_[None, :]
    tl.debug_barrier()
    p_v = v + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]
    b_v = tl.load(p_v, mask=mask_bv)
    b_dk += b_do * b_beta[:, None]
    b_dbeta = tl.sum(b_do * b_k, axis=1)
    b_dbeta += tl.sum(b_dv * b_v, axis=1)
    b_v = None

    p_o = o + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :]
    p_o2 = o2 + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]
    b_o = tl.load(p_o, mask=mask_bk)
    b_o2 = tl.load(p_o2, mask=mask_bv)

    dA = -tl.dot(b_do.to(b_o.dtype), tl.trans(b_o), allow_tf32=False)
    dA -= tl.dot(b_dv.to(b_o2.dtype), tl.trans(b_o2).to(b_o.dtype),
                 allow_tf32=False)
    dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], dA, 0)
    b_dv *= b_beta[:, None]
    p_dv = dv + i_bh * T * DV + (i_t * BT + tl.arange(0, BT)[:, None]) * DV + tl.arange(0, BV)[None, :]
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_bv)

    b_dbeta += tl.sum(dA * tl.dot(b_k, tl.trans(b_k), allow_tf32=False), axis=1)
    dA = dA * b_beta[:, None]
    b_dk += tl.dot(tl.trans(dA.to(b_k.dtype)), b_k, allow_tf32=False)
    b_dk += tl.dot(dA.to(b_k.dtype), b_k, allow_tf32=False)
    p_dk = dk + i_bh * T * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :]
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_bk)
    p_dbeta = dbeta + i_bh * T + i_t * BT + tl.arange(0, BT)
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), mask=mask_bt)


def fwd_prepare_wy_repr(k, v, beta, chunk_size):
    c = chunk_size
    b, h, l, d_k = k.shape
    d_v = v.shape[-1]
    v_new = torch.empty_like(v)
    o_cumdecay = torch.empty_like(k)
    BT = c
    NT = triton.cdiv(l, c)
    BK = triton.next_power_of_2(d_k)
    BV = triton.next_power_of_2(d_v)
    fwd_prepare_wy_repr_kernel[(NT, b*h)](
        k, v, beta, o_cumdecay, v_new,
        NT, d_k, d_v, l, BT, BK, BV
    )
    return o_cumdecay, v_new


def bwd_prepare_wy_repr(k, v, beta, o_cumdecay, v_new, do, do2, chunk_size):
    b, h, l, d_k = do.shape
    d_v = v.shape[-1]
    BK = triton.next_power_of_2(d_k)
    BV = triton.next_power_of_2(d_v)
    c = chunk_size
    BK = d_k
    NT = triton.cdiv(l, c)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.zeros_like(beta)
    bwd_prepare_wy_repr_kernel[(NT, b*h)](
        k, v, beta,
        o_cumdecay, v_new, do, do2,
        dk, dv, dbeta,
        NT, d_k, d_v, l, chunk_size, BK, BV
    )
    return dk, dv, dbeta


class WYRepresentationPrepration(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, k, v, beta, chunk_size):
        o_cumdecay, v_new = fwd_prepare_wy_repr(k, v, beta, chunk_size)
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(k.to(v), v, beta, o_cumdecay, v_new)
        return o_cumdecay, v_new

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, do2):
        k, v, beta, o_cumdecay, v_new = ctx.saved_tensors
        dk, dv, dbeta = bwd_prepare_wy_repr(k, v, beta, o_cumdecay, v_new, do, do2, ctx.chunk_size)
        return dk, dv, dbeta, None


prepare_wy_repr = WYRepresentationPrepration.apply


def naive(k, v, beta, chunk_size):
    l = k.shape[2]
    l_new = triton.next_power_of_2(l)
    # pad k, v, beta
    k = torch.cat([k, torch.zeros_like(k)[:, :, :l_new-l, :]], dim=2)
    v = torch.cat([v, torch.zeros_like(v)[:, :, :l_new-l, :]], dim=2)
    beta = torch.cat([beta, torch.zeros_like(beta)[:, :, :l_new-l]], dim=2)

    k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), (k, v))
    # k = torch.nn.functional.normalize(k, dim=-1, p=2)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device), diagonal=0)
    k_beta = k * beta[..., None]
    v = v * beta[..., None]
    attn = (k @ k.transpose(-1, -2)).masked_fill_(mask, 0)
    attn = attn * beta[..., None]
    x = attn @ v

    o = torch.zeros_like(k)
    o2 = torch.zeros_like(v)

    o[..., 0, :] = k_beta[..., 0, :].clone()
    o2[..., 0, :] = x[..., 0, :].clone()
    for i in range(1, chunk_size):
        o_i = (o[..., :i, :]).clone()
        o[..., i, :] = -(attn[..., i, :i, None] * o_i).sum(3) + k_beta[..., i, :]
        o2_i = (o2[..., :i, :]).clone()
        o2[..., i, :] = -(attn[..., i, :i, None] * o2_i).sum(3) + x[..., i, :]
    return map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d')[:, :, :l], (o, v-o2))


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    seq_len = 125
    b = 128
    h = 4
    k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, 16), dim=-1, p=2)
    v = torch.randn(b, h, seq_len, 32) * 10
    beta = torch.rand(b, h, seq_len).sigmoid()
    require_grad = True
    k, v, beta = map(lambda x: x.cuda().requires_grad_(require_grad), (k, v, beta))
    do = torch.rand_like(k)
    do2 = torch.rand_like(v)

    o1, o2 = naive(k.clone(), v.clone(), beta.clone(), 32)
    if require_grad:
        o1.backward(do, retain_graph=True)
        o2.backward(do2, retain_graph=True)

        k_grad2, v_grad2, beta_grad2 = k.grad, v.grad, beta.grad
        k.grad = v.grad = beta.grad = None

    o3, o4 = prepare_wy_repr(k.clone(), v.clone(), beta.clone(), 32)
    print((o1-o3).abs().max())
    print((o2-o4).abs().max())
    
    if require_grad:
        o3.backward(do, retain_graph=True)
        o4.backward(do2, retain_graph=True)
        k_grad, v_grad, beta_grad = k.grad, v.grad, beta.grad
        print((k_grad2-k_grad).abs().max())
        print((v_grad2-v_grad).abs().max())
        print((beta_grad2-beta_grad).abs().max())
    breakpoint()

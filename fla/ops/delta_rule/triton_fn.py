import triton 
import torch
import triton.language as tl
from einops import rearrange
from fla.ops.utils import contiguous
from packaging import version
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def fwd_prepare_wy_repr(A, x, k, cumsum, cumdecay,
                                  NT, DK,
                                  BT: tl.constexpr,
                                  BK: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) 
    p_x = x + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_k = k + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    S = tl.load(p_x).to(tl.float32)
    p_A = A + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT)
    S_cumdecay = tl.load(p_k).to(tl.float32)
    for i in range(BT):
        attn = tl.load(p_A) 
        mask = tl.arange(0, BT) < i
        attn = tl.where(mask, attn, 0)
        new = tl.sum(attn[:, None] * S, axis=0)
        new_cumdecay = tl.sum(attn[:, None] * S_cumdecay, axis=0)
        mask = tl.arange(0, BT) == i
        S = tl.where(mask[:, None], S - new[None, :], S)
        S_cumdecay = tl.where(mask[:, None], S_cumdecay - new_cumdecay[None, :], S_cumdecay)
        p_A += BT
    p_cumsum = cumsum + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0, BT)[:,  None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_cumsum, S)
    p_cumdecay = cumdecay + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0, BT)[:,  None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK   
    tl.store(p_cumdecay, S_cumdecay)


@triton.jit
def bwd_prepare_wy_repr(A, cumsum, cumdecay,
                        d_cumsum, d_cumdecay, dA, 
                        NT, DK,
                        BT: tl.constexpr,
                        BK: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) 
    p_dcumsum = d_cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_dcumdecay = d_cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_cumsum = cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_cumdecay = cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    
    o = tl.load(p_cumsum).to(tl.float32)
    o2 = tl.load(p_cumdecay).to(tl.float32)
    do = tl.load(p_dcumsum).to(tl.float32)
    do2 = tl.load(p_dcumdecay).to(tl.float32)

    p_A = A + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT) + (BT - 1) * BT
    p_dA = dA + i_bh * NT * BT * BT + i_t * BT * BT + tl.arange(0, BT) + (BT - 1) * BT
    # from the last to the first
    for i in range(BT-1, -1, -1):
        attn = tl.load(p_A) 
        mask = tl.arange(0, BT) < i
        attn = tl.where(mask, attn, 0)
        mask2 = tl.arange(0, BT) == i
        do_ = tl.sum(tl.where(mask2[:, None], do, 0), axis=0)
        do2_ = tl.sum(tl.where(mask2[:, None], do2, 0), axis=0)
        dA_ = tl.where(mask[:, None], o, 0) * do_[None, :] +  tl.where(mask[:, None], o2, 0)  * do2_[None, :]
        dA_ = tl.sum(dA_, axis=1)
        tl.store(p_dA, -dA_)  

        do = do - attn[:, None] * do_[None, :]
        do2 = do2 - attn[:, None] * do2_[None, :]
        p_A -= BT
        p_dA -= BT
    
    p_dcumsum = d_cumsum + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_dcumdecay = d_cumdecay + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_dcumsum, do)
    tl.store(p_dcumdecay, do2)


class WYRepresentationPrepration(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, A, x, k):
        b, h, n, c, d_k = x.shape 
        o_cumsum = torch.empty_like(x)
        o_cumdecay = torch.empty_like(x)
        BT = c
        # assert d_k % 32 == 0
        BK = d_k
        NT = n
        NK = triton.cdiv(d_k, BK)
        fwd_prepare_wy_repr[(NK,NT,b*h)](
        A, x, k, o_cumsum, o_cumdecay, 
        NT, d_k, BT, BK, num_warps=1, num_stages=4
        )
        ctx.save_for_backward(A, o_cumsum, o_cumdecay)
        return o_cumsum, o_cumdecay
    
    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, do2):
        A, o, o2 = ctx.saved_tensors
        b, h, n, c, d_k = o.shape 
        dA = torch.empty_like(A)
        BT = c
        BK = d_k
        NT = n
        NK = triton.cdiv(d_k, BK)
        bwd_prepare_wy_repr[(NK,NT,b*h)](
            A, o, o2, do, do2, dA,
            NT, d_k, BT, BK, num_warps=4, num_stages=4
        )
        return dA, do, do2

prepare_wy_repr = WYRepresentationPrepration.apply
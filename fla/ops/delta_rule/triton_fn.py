import triton 
import torch
import triton.language as tl
from einops import rearrange
from fla.ops.utils import contiguous
from packaging import version
from torch.cuda.amp import custom_bwd, custom_fwd

# Inspired by "THE WY REPRESENTATION FOR PRODUCTS OF HOUSEHOLDER MATRICES" https://epubs.siam.org/doi/pdf/10.1137/0908009
# o: cumprod
# o2: cumprodsum
@triton.jit
def fwd_prepare_wy_repr(k, v, beta, o, o2,
                        NT, DK, 
                        BT: tl.constexpr,
                        BK: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) 
    p_k = k + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_v = v + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_beta = beta + i_bh * NT * BT + i_t * BT + tl.arange(0, BT)
    b_k, b_v, b_beta = tl.load(p_k), tl.load(p_v), tl.load(p_beta)
    
    b_beta = b_beta.to(tl.float32)
    b_v = (b_v * b_beta[:, None]).to(b_v.dtype)
    A = tl.dot(b_k, tl.trans(b_k), allow_tf32=False) * b_beta[:, None]
    A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], A, 0)
    b_o = b_k.to(tl.float32) * b_beta[:, None]
    b_o2 = tl.dot(A.to(b_k.dtype), b_v, allow_tf32=False)

    for i in range(BT):
        # trick to retrieve attn
        mask = tl.arange(0, BT) == i
        # mask2 = tl.arange(0, BT) < i
        attn = tl.sum(tl.where(mask[:, None], A, 0), axis=0)
        # attn = tl.where(mask2, attn, 0)
        new_o = tl.sum(attn[:, None] * b_o, axis=0)
        new_o2 = tl.sum(attn[:, None] * b_o2, axis=0)
        b_o = tl.where(mask[:, None], b_o - new_o[None, :], b_o)
        b_o2 = tl.where(mask[:, None], b_o2 - new_o2[None, :], b_o2)
    
    p_o = o + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0, BT)[:,  None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK   
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
    p_o2 = o2 + i_bh * BT * NT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_o2, (b_v - b_o2).to(p_o2.dtype.element_ty))


@triton.jit
def bwd_prepare_wy_repr(k, v, beta, 
                        o, o2, do, do2,
                        dk, dv, dbeta,
                        NT, DK,
                        BT: tl.constexpr,
                        BK: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) 
    p_k = k + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_do = do + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_do2 = do2 + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK

    p_beta = beta + i_bh * NT * BT + i_t * BT + tl.arange(0, BT)
    b_k, b_beta = tl.load(p_k), tl.load(p_beta)
    b_beta = b_beta.to(tl.float32)
    A = tl.dot(b_k, tl.trans(b_k), allow_tf32=False) * b_beta[:, None]
    A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], A, 0)
    b_do = tl.load(p_do).to(tl.float32)
    b_do2 = -tl.load(p_do2).to(tl.float32)
    dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv -= b_do2 
    for i in range(BT-1, -1, -1):
        mask = tl.arange(0, BT) == i
        attn = tl.sum(tl.where(mask[:, None], A, 0), axis=0)
        do_ = tl.sum(tl.where(mask[:, None], b_do, 0), axis=0)
        do2_ = tl.sum(tl.where(mask[:, None], b_do2, 0), axis=0)
        mask2 = tl.arange(0, BT) < i
        b_do = b_do - attn[:, None] * do_[None, :]
        b_do2 = b_do2 - attn[:, None] * do2_[None, :]
    tl.debug_barrier()

    p_v = v + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    b_v = tl.load(p_v) 
    b_dk += b_do * b_beta[:, None] 
    b_dbeta = tl.sum(b_do * b_k, axis=1)
    
    # compute dv

    b_dv += tl.dot(tl.trans(A.to(b_k.dtype)), b_do2.to(b_k.dtype), allow_tf32=False)
    b_dbeta += tl.sum(b_dv * b_v, axis=1)
    b_v *= b_beta[:, None]
    b_dv *= b_beta[:, None]
    p_dv = dv + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))

    p_o = o + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    p_o2 = o2 + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    b_o = tl.load(p_o)
    b_o2 = tl.load(p_o2)

    dA = tl.dot(b_do.to(b_o.dtype), tl.trans(b_o), allow_tf32=False)
    dA += tl.dot(b_do2.to(b_o2.dtype), tl.trans(b_v - b_o2).to(b_o.dtype), allow_tf32=False)
    dA = tl.where(
        tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :],
        tl.dot(b_do2.to(b_v.dtype), tl.trans(b_v), allow_tf32=False) - dA, 
        0
    )

    b_dbeta += tl.sum(dA * tl.dot(b_k, tl.trans(b_k), allow_tf32=False), axis=1)
    
    dA = dA * b_beta[:, None]
    
    b_dk += tl.dot(tl.trans(dA.to(b_k.dtype)), b_k, allow_tf32=False)
    
    b_dk += tl.dot(dA.to(b_k.dtype), b_k, allow_tf32=False)

    p_dk = dk + i_bh * NT * BT * DK + (i_t * BT + tl.arange(0, BT)[:, None]) * DK + tl.arange(0, BK)[None, :] + i_k * BK
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))    

    p_dbeta = dbeta + i_bh * NT * BT + i_t * BT + tl.arange(0, BT)
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty))

class WYRepresentationPrepration(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, k, v, beta, chunk_size):
        c = chunk_size
        b, h, l, d_k = k.shape
        # b, h, n, c, d_k = x.shape 
        v_new = torch.empty_like(k).fill_(float("-inf"))
        o_cumdecay = torch.empty_like(k).fill_(float("-inf"))
        BT = ctx.BT = c
        BK = d_k
        NT = l // c
        NK = triton.cdiv(d_k, BK)
        assert NK == 1
        fwd_prepare_wy_repr[(NK, NT, b*h)](
        k, v, beta, o_cumdecay, v_new,
        NT, d_k, BT, BK, num_warps=2, num_stages=4
        )
        ctx.save_for_backward(k, v, beta, o_cumdecay.clone(), v_new.clone())
        return o_cumdecay, v_new
    
    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, do2):
        k, v, beta, o_cumdecay, v_new = ctx.saved_tensors
        b, h, l, d_k = do.shape 
        c = BT = ctx.BT
        BK = d_k
        NT = l // c
        NK = triton.cdiv(d_k, BK)
        assert NK == 1
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dbeta = torch.zeros_like(beta)
        dA = torch.zeros([BT, BT], dtype=torch.float32)
        bwd_prepare_wy_repr[(NK, NT, b*h)](
            k, v, beta,
            o_cumdecay, v_new, do, do2,
            dk, dv, dbeta,
            NT, d_k, BT, BK, num_warps=2, num_stages=2
        )
        return dk, dv, dbeta, None

prepare_wy_repr = WYRepresentationPrepration.apply


def naive(k, v, beta, chunk_size):
    k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), (k, v))
    beta = rearrange(beta, 'b h (n c) -> b h n c', c = chunk_size)
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
    return map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), (o, v-o2))


if __name__ == "__main__":
    seq_len = 32
    b = 2
    h = 4
    k = torch.randn(b, h, seq_len, 64) / 10
    v = torch.randn(b, h, seq_len, 64)  
    beta = torch.rand(b, h, seq_len).sigmoid()
    # beta = torch.ones(b, h, seq_len)
    # beta = torch.ones(b, h, seq_len)
    require_grad = True
    k, v, beta = map(lambda x: x.cuda().to(torch.bfloat16).requires_grad_(require_grad), (k, v, beta))
    do = torch.rand_like(v)
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

        

    





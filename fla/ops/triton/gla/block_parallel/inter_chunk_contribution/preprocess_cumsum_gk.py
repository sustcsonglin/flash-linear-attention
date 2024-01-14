from fla.ops.triton.utils import contiguous
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import time
from torch.cuda.amp import custom_bwd, custom_fwd

# def stable_logsigmoid(x):
#     # Use the identity log(sigmoid(x)) = -log(1 + exp(-x))
#     # This is stable for large negative values of x
#     neg_abs_x = -torch.abs(x)
#     return torch.where(x < 0, x, neg_abs_x) - torch.log1p(torch.exp(neg_abs_x))


@triton.jit
def _fwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum,
    Q_exp, K_reduce, GK_last_exp,
    NUM_CHUNK, L,
    D_MODEL_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    Q_exp_ptr = Q_exp + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_last_exp_ptr = GK_last_exp + offset_bh * NUM_CHUNK * \
        D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K)

    cumsum = tl.zeros([D_MODEL_K], dtype=tl.float32)

    for _ in range(CHUNK_SIZE):
        gk = tl.load(GK_ptr).to(tl.float32)
        cumsum += gk
        tl.store(GK_cumsum_ptr, cumsum.to(GK_cumsum_ptr.dtype.element_ty))
        cumsum_exp = tl.exp(cumsum)
        q = tl.load(Q_ptr)
        q_exp = q * cumsum_exp
        tl.store(Q_exp_ptr, q_exp)
        Q_ptr += D_MODEL_K
        Q_exp_ptr += D_MODEL_K
        GK_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K

    tl.store(GK_last_exp_ptr, tl.exp(cumsum).to(
        GK_last_exp_ptr.dtype.element_ty))

    tl.debug_barrier()

    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_reduce_ptr = K_reduce + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    for _ in range(CHUNK_SIZE):
        gk_cumsum = tl.load(GK_cumsum_ptr)
        k = tl.load(K_ptr)
        k_reduce = k * tl.exp(cumsum - gk_cumsum)
        tl.store(K_reduce_ptr, k_reduce.to(K_reduce_ptr.dtype.element_ty))

        K_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
        K_reduce_ptr += D_MODEL_K


@triton.jit
def _bwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum,

    DQ_exp, DK_reduce, DGK_last_exp, DGK_cumsum,

    DQ, DK, DGK,

    NUM_CHUNK, L,
    D_MODEL_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    DQ_ptr = DQ + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_ptr = DK + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DQ_exp_ptr = DQ_exp + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_reduce_ptr = DK_reduce + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_cumsum_ptr = DGK_cumsum + offset_bh * L * D_MODEL_K + \
        offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * \
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    D_GK_last_exp_ptr = DGK_last_exp + offset_bh * NUM_CHUNK * \
        D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K)
    #
    cumsum_gradient = tl.zeros([D_MODEL_K], dtype=tl.float32)
    grad_gk_last = tl.zeros([D_MODEL_K], dtype=tl.float32)

    gk_last = tl.load(GK_cumsum_ptr + (CHUNK_SIZE - 1)
                      * D_MODEL_K).to(tl.float32)
    cumsum_gradient += tl.load(D_GK_last_exp_ptr) * tl.exp(gk_last)

    GK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    GK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    Q_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    K_ptr += (CHUNK_SIZE - 1) * D_MODEL_K

    DQ_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K

    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gk_cs = tl.load(GK_cumsum_ptr).to(tl.float32)
        k = tl.load(K_ptr).to(tl.float32)
        grad_k = tl.exp(gk_last - gk_cs) * \
            tl.load(DK_reduce_ptr).to(tl.float32)
        tl.store(DK_ptr, grad_k.to(DK_ptr.dtype.element_ty))
        grad_k *= k
        cumsum_gradient -= grad_k
        grad_gk_last += grad_k

        q = tl.load(Q_ptr).to(tl.float32)
        grad_q = tl.exp(gk_cs) * tl.load(DQ_exp_ptr)
        tl.store(DQ_ptr, grad_q.to(DK_ptr.dtype.element_ty))
        cumsum_gradient += grad_q * q.to(tl.float32)

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGK_cumsum_ptr).to(tl.float32)

        tl.store(DGK_ptr, cumsum_gradient.to(DGK_ptr.dtype.element_ty))

        Q_ptr -= D_MODEL_K
        DQ_exp_ptr -= D_MODEL_K
        K_ptr -= D_MODEL_K
        DK_reduce_ptr -= D_MODEL_K
        GK_cumsum_ptr -= D_MODEL_K
        DGK_cumsum_ptr -= D_MODEL_K
        DQ_ptr -= D_MODEL_K
        DK_ptr -= D_MODEL_K
        DGK_ptr -= D_MODEL_K

    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * \
        D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * \
        D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K

    # tl.store(D_GK_last_exp_ptr, cumsum_gradient)

    # seems stupid. just workaround some compiler bugs.
    grad_gk_last = grad_gk_last + 0.
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgk = tl.load(DGK_ptr).to(tl.float32)
        dgk += grad_gk_last
        tl.store(DGK_ptr, dgk.to(DGK_ptr.dtype.element_ty))
        DGK_ptr -= D_MODEL_K
        GK_ptr -= D_MODEL_K


class PreprocessCumSum_GK(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, gk):
        B, H, NUM_CHUNK, CHUNK_SIZE, D = q.shape

        D_k = k.shape[-1]
        grid = (B * H, NUM_CHUNK)
        ctx.grid = grid

        k_reduce = torch.empty_like(k)

        q_exp = torch.empty_like(q)

        gk_cumsum = torch.empty_like(gk)

        gk_last_exp = torch.empty_like(gk[:, :, :, 0], dtype=torch.float32)

        _fwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum,
            q_exp, k_reduce, gk_last_exp,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK=NUM_CHUNK, L=CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )

        ctx.grid = grid
        ctx.save_for_backward(q, k, gk, gk_cumsum)

        return gk_cumsum, k_reduce, q_exp,  gk_last_exp

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, dgk_cumsum, dk_reduce, dq_exp, dgk_last_exp):
        q, k, gk, gk_cumsum = ctx.saved_tensors
        grid = ctx.grid

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dgk = torch.empty_like(gk)

        B, H, NUM_CHUNK, CHUNK_SIZE, D_k = q.shape

        # D_v = v.shape[-1]

        _bwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum,
            dq_exp, dk_reduce, dgk_last_exp, dgk_cumsum,
            dq, dk, dgk,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK=NUM_CHUNK, L=CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )

        return dq.to(q.dtype), dk.to(k.dtype), dgk.to(gk.dtype), None, None, None

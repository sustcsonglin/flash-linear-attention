from fla.ops.triton.utils import contiguous
import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_apply_decay_qk(
    q, k, scale,
    q_decay, k_decay,
    s_qk_h, s_qk_t, s_qk_d,
    H, T, DK,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h + i_c * BT * s_qk_t,
                            (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h + i_c * BT * s_qk_t,
                            (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    o_i = tl.arange(0, BT)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_o, d_h = tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)
    b_q = tl.load(p_q, boundary_check=(1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(1)).to(tl.float32)
    b_q *= d_o[:, None] * scale
    b_k *= d_h[:, None]
    p_q_decay = tl.make_block_ptr(q_decay + i_bh * s_qk_h + i_c * BT * s_qk_t,
                                  (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k_decay = tl.make_block_ptr(k_decay + i_bh * s_qk_h + i_c * BT * s_qk_t,
                                  (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_q_decay, b_q.to(p_q_decay.dtype.element_ty), boundary_check=(1))
    tl.store(p_k_decay, b_k.to(p_k_decay.dtype.element_ty), boundary_check=(1))


@triton.jit
def _bwd_apply_decay_qk(
    dq, dk, scale,
    s_qk_h, s_qk_t, s_qk_d,
    H, T, DK,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h + i_c * BT * s_qk_t,
                             (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h + i_c * BT * s_qk_t,
                             (BT, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    o_i = tl.arange(0, BT)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_o, d_h = tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)
    b_dq = tl.load(p_dq, boundary_check=(1)).to(tl.float32)
    b_dk = tl.load(p_dk, boundary_check=(1)).to(tl.float32)
    b_dq *= d_o[:, None] * scale
    b_dk *= d_h[:, None]
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(1))


class PrepareRetentionQK(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, k):
        q_decay = torch.empty_like(q)
        k_decay = torch.empty_like(k)
        batch_size, n_heads, n_chunks, chunk_size, d_head_k = q.shape
        BK = min(32, triton.next_power_of_2(d_head_k))

        grid = (triton.cdiv(d_head_k, BK), n_chunks, batch_size * n_heads)
        scale = q.shape[-1]**-0.5

        _fwd_apply_decay_qk[grid](
            q, k, scale,
            q_decay, k_decay,
            q.stride(1), q.stride(3), q.stride(4),
            n_heads, chunk_size, d_head_k,
            chunk_size, BK, num_warps=8
        )
        return q_decay, k_decay

    @staticmethod
    @contiguous
    def backward(ctx, dq, dk):
        batch_size, n_heads, n_chunks, chunk_size, d_head_k = dq.shape
        BK = min(32, triton.next_power_of_2(d_head_k))
        grid = (triton.cdiv(d_head_k, BK), n_chunks, batch_size * n_heads)
        _bwd_apply_decay_qk[grid](
            dq, dk, d_head_k ** -0.5,
            dq.stride(1), dq.stride(3), dq.stride(4),
            n_heads, chunk_size, d_head_k,
            chunk_size, BK, num_warps=8
        )
        return dq, dk, None


@triton.jit
def _fwd_recurrence(
    S,
    O,
    NUM_CHUNK, H, CHUNK_SIZE,
    DK, DV,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(CHUNK_SIZE * b_b)
    i_k = tl.program_id(1)
    i_v = tl.program_id(2)
    mask = (i_k * BK + tl.arange(0, BK) <
            DK)[:, None] & (i_v * BV + tl.arange(0, BV) < DV)[None, :]
    S = S + i_bh * NUM_CHUNK * DK * DV + i_k * BK * DV + \
        tl.arange(0, BK)[:, None] * DV + i_v * BV + tl.arange(0, BV)[None, :]
    O = O + i_bh * NUM_CHUNK * DK * DV + i_k * BK * DV + \
        tl.arange(0, BK)[:, None] * DV + i_v * BV + \
        tl.arange(0, BV)[None, :] + DK * DV
    acc = tl.zeros([BK, BV], dtype=tl.float32)
    acc += tl.load(S, mask=mask, other=0)
    S += DK * DV
    tl.store(O, acc.to(O.dtype.element_ty), mask=mask)
    O += DK * DV
    for i in range(NUM_CHUNK-2):
        S_i = tl.load(S, mask=mask, other=0)
        acc = d_b * acc + S_i
        tl.store(O, acc.to(O.dtype.element_ty), mask=mask)
        S += DK * DV
        O += DK * DV


@triton.jit
def _bwd_recurrence(
    S,
    DS,
    H,
    NUM_CHUNK, CHUNK_SIZE,
    DK, DV,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_k = tl.program_id(1)
    i_v = tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(CHUNK_SIZE * b_b)

    # skip the last chunk because it is never used
    S = S + i_bh * NUM_CHUNK * DK * DV + i_k * BK * DV + \
        tl.arange(0, BK)[:, None] * DV + i_v * BV + \
        tl.arange(0, BV)[None, :] + (NUM_CHUNK - 2) * DK * DV

    # start from the last chunk
    DS = DS + i_bh * NUM_CHUNK * DK * DV + i_k * BK * DV + \
        tl.arange(0, BK)[:, None] * DV + i_v * BV + \
        tl.arange(0, BV)[None, :] + (NUM_CHUNK - 1) * DK * DV
    Dacc = tl.zeros([BK, BV], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK) <
            DK)[:, None] & (i_v * BV + tl.arange(0, BV) < DV)[None, :]
    # ignore the first chunk
    for i in range(NUM_CHUNK - 1):
        DS_i = tl.load(DS, mask=mask, other=0)
        Dacc += DS_i
        tl.store(S, Dacc.to(S.dtype.element_ty), mask=mask)
        Dacc *= d_b
        S -= DK * DV
        DS -= DK * DV


class ChunkStateScanRetention(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, to_add, chunk_size=256):
        B, H, N, D_k, D_v = to_add.shape
        output = torch.empty_like(to_add)
        BK = min(32, triton.next_power_of_2(D_k))
        BV = min(32, triton.next_power_of_2(D_v))
        NK, NV = triton.cdiv(D_k, BK), triton.cdiv(D_v, BV)
        grid = (B*H, NK, NV)
        ctx.grid = grid
        _fwd_recurrence[grid](
            to_add,
            output,
            DK=D_k, DV=D_v,
            NUM_CHUNK=N, H=H, CHUNK_SIZE=chunk_size,
            BK=BK, BV=BV, num_warps=1
        )
        output[:, :, 0] = 0
        ctx.save_for_backward(output)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()
        output, = ctx.saved_tensors
        B, H, N, D_k, D_v = output.shape
        chunk_size = ctx.chunk_size
        num_block = N
        BK = min(32, triton.next_power_of_2(D_k))
        BV = min(32, triton.next_power_of_2(D_v))
        NK, NV = triton.cdiv(D_k, BK), triton.cdiv(D_v, BV)
        grid = (B*H, NK, NV)
        _bwd_recurrence[grid](
            output,
            DO,
            NUM_CHUNK=num_block,
            DK=D_k,
            DV=D_v, H=H, CHUNK_SIZE=chunk_size,
            BK=BK, BV=BV, num_warps=1
        )
        output[:, :, -1] = 0
        return output, None


apply_decay_qk_retention = PrepareRetentionQK.apply
scan_memory_state_retention = ChunkStateScanRetention.apply

# -*- coding: utf-8 -*-

# Code adapted from
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536   # the best size we found by manually tuning


@triton.jit
def cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr:
        Pointer to input tensor.
    X_stride (int):
        The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int):
        The stride of the target tensor.
    loss_ptr:
        Pointer to tensor to store the loss.
    loss_stride (int):
        The stride of the loss tensor.
    n_cols (int):
        The number of columns in the input tensor.
    n_non_ignore (int):
        The number of non-ignored elements in the batch.
    ignore_index (int):
        The index to ignore in the target.
    label_smoothing (float):
        The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    BLOCK_SIZE (int):
        The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y)  # we need to store the original value of X_y for the loss calculation

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 4. [Online softmax] second pass: calculate the gradients
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # N is the number of non ignored elements in the batch
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = -(ori_X_y - m - tl.log(d))

    # Orginal loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper:
    # https://arxiv.org/pdf/1512.00567
    # pytorch:
    # https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    X_y = tl.load(X_ptr + y)
    X_y += -(1 - label_smoothing) / (n_non_ignore)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


@triton.jit
def element_mul_kernel(
    x,
    g,
    s_x,
    N: tl.constexpr,
    B: tl.constexpr
):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    s_x (int):
        The stride of the input tensor.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    i_x = tl.program_id(0).to(tl.int64)

    # Locate the start index
    x += i_x * s_x

    # Load the gradient output value
    b_g = tl.load(g)

    # Perform the element-wise multiplication
    for i in range(0, N, B):
        o_x = i + tl.arange(0, B)
        b_x = tl.load(x + o_x, mask=o_x < N)
        tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
):
    device = x.device

    # inputs have shape: [BT, H]
    # materialized activations will have shape: [BT, V]
    # the increase in memory = [BT, V]
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for example: if we were to achieve the same memory consumption as [BT, H], then the chunk size should be:
    # inc_factor = (V+H-1)//H, C = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, C = 2048
    BT, H, V = *x.shape, weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    dw = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    dx = torch.zeros_like(x, device=device)
    db = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # NOTE: skip .item() here to avoid CUDA synchronization
    total_n_non_ignore = (target != ignore_index).sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        # [C, N]
        x_chunk = x[start_idx:end_idx]
        # when doing matmul, use the original precision
        # [C, V]
        logits_chunk = F.linear(x_chunk, weight, bias).contiguous()
        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        n_non_ignore = (target_chunk != ignore_index).sum().item()

        # ensure x and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
        # thus dx[start_idx: end_idx] should be of shape: chunk_size x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.
        grad_logits_chunk = logits_chunk * n_non_ignore / total_n_non_ignore  # chunk_size x V
        dx[start_idx:end_idx] = grad_logits_chunk @ weight

        if dw is not None:
            torch.addmm(
                input=dw,
                mat1=logits_chunk.t(),
                mat2=x_chunk,
                out=dw,
                alpha=n_non_ignore / total_n_non_ignore,
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=db,
                other=logits_chunk.sum(dim=0),
                out=db,
                alpha=n_non_ignore / total_n_non_ignore,
            )

    loss = torch.sum(loss_1d) / total_n_non_ignore
    return loss, dx, dw, db


def fused_linear_cross_entropy_backward(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor
):
    # If cross entropy is the last layer, do is 1.0. Skip the mul to save time
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = dx.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            x=dx,
            g=do,
            s_x=dx.stride(-2),
            N=H,
            B=BLOCK_SIZE,
            num_warps=32,
        )

        # handle dw
        if dw is not None:
            V, H = dw.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                x=dw,
                g=do,
                s_x=dw.stride(-2),
                N=H,
                B=BLOCK_SIZE,
                num_warps=32,
            )

        if db is not None:
            V = db.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                x=db,
                g=do,
                s_x=db.stride(-1),
                N=1,
                B=BLOCK_SIZE,
                num_warps=32,
            )
    return dx, dw, db


class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the x and target
        for the backward pass.

        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index:
            the index to ignore in the target.
        label_smoothing:
            the amount of smoothing when computing the loss, where 0.0 means no smoothing.
        """
        loss, dx, dw, db = fused_linear_cross_entropy_forward(
            x,
            target,
            weight,
            bias,
            ignore_index,
            label_smoothing
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            dx.detach(),
            dw.detach() if weight is not None else None,
            db.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None


def fused_linear_cross_entropy_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
    Returns:
        losses: [batch,], float
        z_losses: [batch,], float
    """
    return FusedLinearCrossEntropyFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        label_smoothing,
    )


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        """
        Arguments:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """
        Arguments:
            x (torch.Tensor): [batch_size * seq_len, hidden_size]
            target (torch.LongTensor): [batch_size * seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_linear_cross_entropy_loss(
            x,
            target,
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss

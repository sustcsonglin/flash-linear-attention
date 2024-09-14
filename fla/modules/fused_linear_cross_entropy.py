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
MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def cross_entropy_kernel(
    x,
    target,
    loss,
    s_x,
    s_target,
    s_loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        x:
            Pointer to input tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        s_x (int):
            The stride of the input tensor.
        s_target (int):
            The stride of the target tensor.
        s_loss (int):
            The stride of the loss tensor.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, i_n * stride will overflow out of int32, so we convert to int64
    i_n = tl.program_id(0).to(tl.int64)

    # 1. Load target first because if the target is ignore_index, we can return right away
    target += i_n * s_target
    y = tl.load(target)

    # 2. locate the start index
    x += i_n * s_x

    if y == ignore_index:
        # set all x as 0
        for i in range(0, V, BV):
            o_x = i + tl.arange(0, BV)
            tl.store(x + o_x, 0.0, mask=o_x < V)
        return

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float('-inf')  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    b_x_t = tl.load(x + y)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    b_z = 0.0
    eps = label_smoothing / V

    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_x = tl.load(x + o_x, mask=o_x < V, other=float('-inf'))
        b_m = tl.max(b_x)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            b_z += tl.sum(tl.where(o_x < V, -eps * b_x, 0.0))
        m_new = tl.maximum(m, b_m)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(b_x - m_new))
        m = m_new

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_x = tl.load(x + o_x, mask=o_x < V, other=float('-inf'))
        b_x = (tl.exp(b_x - m) / d - eps)
        if reduction == "mean":
            b_x = b_x / total
        tl.store(x + o_x, b_x, mask=o_x < V)

    # We need tl.debug_barrier() to ensure the new result of x is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss
    # loss = log (softmax(b_x_t)) = log ((e ^ (b_x_t - max(X)) / sum(e ^ (X - max(X))))
    #      = (b_x_t - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(b_x_t)) without overflow
    b_loss = -(b_x_t - m - tl.log(d))

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
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * (m + tl.log(d)))

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    b_x_t = tl.load(x + y)

    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == 'mean':
        b_loss = b_loss / total
        b_x_t += (label_smoothing - 1) / total
    else:
        b_x_t += label_smoothing - 1

    tl.store(loss + i_n * s_loss, b_loss)
    tl.store(x + y, b_x_t)


@triton.jit
def elementwise_mul_kernel(
    x,
    g,
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
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    # Load the gradient output value
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
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
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    C = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    NC = triton.cdiv(BT, C)  # (BT + C - 1) // C

    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device) if weight is not None else None
    db = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss = torch.zeros(BT, dtype=torch.float32, device=device)

    total = target.ne(ignore_index).sum().item()

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, BT)
        # [C, N]
        c_x = x[start:end]
        # when doing matmul, use the original precision
        # [C, V]
        c_logits = F.linear(c_x, weight, bias)
        c_target = target[start:end]

        # unreduced loss
        c_loss = loss[start:end]

        # Here we calculate the gradient of c_logits in place so we can save memory.
        cross_entropy_kernel[(c_logits.shape[0],)](
            x=c_logits,
            target=c_target,
            loss=c_loss,
            s_x=c_logits.stride(-2),
            s_target=c_target.stride(-1),
            s_loss=c_loss.stride(-1),
            total=total,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=32
        )

        # gradient of c_logits is computed in-place by the above triton kernel and is of shape: C x V
        # thus dx[start: end] should be of shape: C x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total) to scale the gradients.
        # [C, H]

        dx[start:end] = torch.mm(c_logits, weight)

        if weight is not None:
            torch.addmm(input=dw, mat1=c_logits.t(), mat2=c_x, out=dw)

        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0), out=db)

    loss = loss.sum()
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
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N*H,
            B=B,
            num_warps=32,
        )

        # handle dw
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V*H,
                B=B,
                num_warps=32,
            )

        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[(triton.cdiv(V, B),)](
                x=db,
                g=do,
                N=V,
                B=B,
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
        label_smoothing: float = 0.0,
        reduction: str = "mean"
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
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        """
        loss, dx, dw, db = fused_linear_cross_entropy_forward(
            x,
            target,
            weight,
            bias,
            ignore_index,
            label_smoothing,
            reduction
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
        return dx, None, dw, db, None, None, None


def fused_linear_cross_entropy_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
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
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
    Returns:
        losses: [batch,], float
    """
    return FusedLinearCrossEntropyFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        label_smoothing,
        reduction
    )


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
        """
        super().__init__()

        assert reduction in ["none", "mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """
        Args:
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
            reduction=self.reduction
        )
        return loss

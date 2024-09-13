import torch
import torch.nn.functional as F


def reference_torch(
    x: torch.Tensor,
    x_weight: torch.Tensor,
    target_x: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = "batchmean",
):
    V = x_weight.shape[0]
    logits = F.linear(x, x_weight).view(-1, V)
    target_probs = F.linear(target_x, target_weight).view(-1, V)
    target_probs = F.softmax(target_probs, dim=-1)

    kl_loss = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target_probs,
        reduction=reduction,
    )
    return kl_loss


class ChunkedKLDiv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_weight: torch.Tensor,
        target_x: torch.Tensor,
        target_weight: torch.Tensor,
        reduction: str = "batchmean",
        sp: int = 8,
    ):
        T = x.size(1)
        chunk_size = (T + sp - 1) // sp

        if reduction == "batchmean":
            reduction_factor = T
        elif reduction == "mean":
            reduction_factor = sp
        elif reduction == "sum":
            reduction_factor = 1
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

        kl_loss = 0
        for i in range(sp):
            logits_i = F.linear(
                x[:, i * chunk_size : (i + 1) * chunk_size, :], x_weight
            )
            target_probs_i = F.linear(
                target_x[:, i * chunk_size : (i + 1) * chunk_size, :], target_weight
            )
            log_probs_i = F.log_softmax(logits_i, dim=-1)
            target_probs_i = F.softmax(target_probs_i, dim=-1)

            loss_i = F.kl_div(
                log_probs_i,
                target_probs_i,
                reduction=reduction,
            )

            kl_loss = kl_loss + loss_i

        kl_loss = kl_loss / reduction_factor

        ctx.save_for_backward(x, x_weight, target_x, target_weight)

        ctx.sp = sp
        ctx.reduction = reduction
        return kl_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, x_weight, target_x, target_weight = ctx.saved_tensors
        sp = ctx.sp
        reduction = ctx.reduction

        B, T, _ = x.size()
        V = x_weight.size(0)

        chunk_size = (T + sp - 1) // sp

        if reduction == "batchmean":
            reduction_factor = B * T
        elif reduction == "mean":
            reduction_factor = B * T * V
        elif reduction == "sum":
            reduction_factor = 1

        grad_x = []
        grad_weight = 0
        for i in range(sp):
            chunk_x = x[:, i * chunk_size : (i + 1) * chunk_size, :]
            logits = F.linear(chunk_x, x_weight)
            target_probs = F.linear(
                target_x[:, i * chunk_size : (i + 1) * chunk_size, :], target_weight
            )
            target_probs = F.softmax(target_probs, dim=-1)

            d_logits = -target_probs + torch.softmax(logits, dim=-1)

            d_logits = d_logits / reduction_factor

            grad_x.append(torch.einsum("blv, vh -> blh", d_logits, x_weight))
            grad_weight += torch.einsum("blv, blh -> vh", d_logits, chunk_x)

        grad_x = torch.cat(grad_x, dim=1)

        return grad_output * grad_x, grad_output * grad_weight, None, None, None, None

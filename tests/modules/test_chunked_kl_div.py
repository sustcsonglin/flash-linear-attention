# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import LowMemKLDiv


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("T", [1, 50, 2048, 4096])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ["mean", "batchmean", "sum"])
def test_fused(B: int, T: int, D: int, V: int, reduction: str):
    torch.manual_seed(42)
    x = torch.randn(B, T, D).cuda().requires_grad_()
    x_weight = torch.randn(V, D).cuda().requires_grad_()
    target_x = torch.randn(B, T, D).cuda()
    target_weight = torch.randn(V, D).cuda()

    logits = F.linear(x, x_weight)
    target_probs = F.linear(target_x, target_weight)
    target_probs = F.softmax(target_probs, dim=-1)

    ref = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target_probs,
        reduction=reduction,
    )

    do = torch.randn_like(ref).cuda()

    ref.backward(do)
    ref_d_x, x.grad = x.grad.clone(), None
    ref_d_x_weight, x_weight.grad = x_weight.grad.clone(), None

    chunk = LowMemKLDiv.apply(x, x_weight, target_x, target_weight, reduction)
    chunk.backward(do)
    chunk_d, x.grad = x.grad.clone(), None
    chunk_d_x_weight, x_weight.grad = x_weight.grad.clone(), None

    torch.testing.assert_close(ref, chunk)
    torch.testing.assert_close(ref_d_x, chunk_d)
    torch.testing.assert_close(ref_d_x_weight, chunk_d_x_weight)

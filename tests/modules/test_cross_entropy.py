# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_cross_entropy(B: int, T: int, D: int, V: int, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)
    logits = torch.randn(B * T, V).cuda().to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B * T,)).cuda()

    ref = nn.CrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    do = torch.randn_like(ref).cuda().to(dtype=dtype)

    ref.backward(do)
    ref_d, logits.grad = logits.grad.clone(), None

    tri = FusedCrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    tri.backward(do)
    tri_d, logits.grad = logits.grad.clone(), None

    torch.testing.assert_close(ref, tri)
    torch.testing.assert_close(ref_d, tri_d)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ['mean', 'sum'])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_linear_cross_entropy(B: int, T: int, D: int, V: int, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)

    x = torch.randn(B * T, D).cuda().to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B * T,)).cuda()
    weight = torch.randn(V, D).cuda().to(dtype=dtype).requires_grad_()
    bias = torch.randn(V).cuda().to(dtype=dtype).requires_grad_()

    logits = F.linear(x, weight, bias)
    ref = nn.CrossEntropyLoss(reduction=reduction)(logits, target)
    do = torch.randn_like(ref).cuda().to(dtype=dtype)

    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, weight.grad = weight.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri = FusedLinearCrossEntropyLoss(reduction=reduction)(x, target, weight, bias).to(dtype=dtype)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, weight.grad = weight.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    assert torch.allclose(ref, tri, atol=1e-5),       f" o diff: {torch.abs(ref - tri).max()}"
    assert torch.allclose(ref_dx, tri_dx, atol=1e-5), f"dx diff: {torch.abs(ref_dx - tri_dx).max()}"
    assert torch.allclose(ref_dw, tri_dw, atol=1e-5), f"dw diff: {torch.abs(ref_dw - tri_dw).max()}"
    assert torch.allclose(ref_db, tri_db, atol=1e-5), f"db diff: {torch.abs(ref_db - tri_db).max()}"
    assert torch.allclose(ref_db, tri_db, atol=1e-5), f"db diff: {torch.abs(ref_db - tri_db).max()}"

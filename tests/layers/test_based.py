# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.gla import BasedLinearAttention


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_based(batch_size: int, seq_len: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(batch_size, seq_len, hidden_size).to(dtype).cuda().requires_grad_(True)
    dy = torch.randn(batch_size, seq_len, hidden_size).to(dtype).cuda()
    model = BasedLinearAttention(hidden_size, mode='chunk').to(dtype).cuda()
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = model.forward_reference(x)
    y2.backward(dy)
    assert y.allclose(y2, 0, 1e-4)
    assert x_grad.allclose(x.grad, 0, 1e-4)

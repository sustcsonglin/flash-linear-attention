# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.delta_rule import (chunk_delta_rule, fused_chunk_delta_rule,
                                fused_recurrent_delta_rule)
from fla.ops.delta_rule.naive import delta_rule_recurrence


@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_beta_scalar_vector_equivalence(B: int, H: int, T: int, D: int, dtype: torch.dtype):
    torch.manual_seed(17)
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=dtype), p=2, dim=-1)
    v = torch.randn(B, H, T, D, dtype=dtype)
    beta = torch.rand(B, H, T, D, dtype=dtype).sigmoid()
    q, k, v, beta = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta))
    do = torch.rand_like(v)

    o = delta_rule_recurrence(q.clone(), k.clone(), v.clone(), beta.clone())
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    o2, _ = fused_recurrent_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone())
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    assert o.allclose(o2, rtol=0, atol=2e-5), f"Diff: {torch.abs(o - o2).max()}"
    assert q_grad.allclose(q_grad2, rtol=0, atol=2e-5), f"Diff: {torch.abs(q_grad - q_grad2).max()}"
    assert k_grad.allclose(k_grad2, rtol=0, atol=2e-5), f"Diff: {torch.abs(k_grad - k_grad2).max()}"
    assert v_grad.allclose(v_grad2, rtol=0, atol=2e-5), f"Diff: {torch.abs(v_grad - v_grad2).max()}"
    # FIXME: this gradient does not match when beta a vector. matches when a scalar.
    assert beta_grad.allclose(beta_grad2, rtol=0, atol=1e-3), f"Diff: {torch.abs(beta_grad - beta_grad2).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_fused_equivalence(B: int, H: int, T: int, D: int, dtype: torch.dtype):
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=dtype), p=2, dim=-1)
    v = torch.randn(B, H, T, D, dtype=dtype)
    beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
    q, k, v, beta = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta))
    do = torch.rand_like(v)

    o2, _ = fused_chunk_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone(), 16)
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    o, _ = chunk_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone(), 32)
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    assert o.allclose(o2, 0, 1e-2), f"Diff: {torch.abs(o - o2).max()}"
    assert q_grad.allclose(q_grad2, 0, 1e-2), f"Diff: {torch.abs(q_grad - q_grad2).max()}"
    assert k_grad.allclose(k_grad2, 0, 1e-2), f"Diff: {torch.abs(k_grad - k_grad2).max()}"
    assert v_grad.allclose(v_grad2, 0, 1e-2), f"Diff: {torch.abs(v_grad - v_grad2).max()}"
    assert beta_grad.allclose(beta_grad2, 0, 1e-2), f"Diff: {torch.abs(beta_grad - beta_grad2).max()}"

import time

import torch

from fla.ops.delta_rule import (chunk_delta_rule, fused_chunk_delta_rule,
                                fused_recurrent_linear_attn_delta_rule)


def gen_inputs(b=8, h=8, length=2048, d=256, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    seq_len = length

    k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, d), p=2, dim=-1)
    v = torch.randn(b, h, seq_len, d)
    q = torch.randn(b, h, seq_len, d)
    beta = torch.rand(b, h, seq_len).sigmoid()
    q, k, v, beta = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta))
    do = torch.rand_like(v)
    return q, k, v, beta, do


def test_beta_scalar_vector_equivalence(q, k, v, beta, do):
    # do smaller inputs bc otherwise low precision
    q, k, v, beta, do = map(lambda x: x[:2, :2, :256].detach().clone(), (q, k, v, beta, do))
    q, k, v, beta = map(lambda x: x.requires_grad_(True), (q, k, v, beta))
    # scalar_beta
    o2, _ = fused_recurrent_linear_attn_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone())
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    # make beta a vector
    beta = beta.detach().clone().unsqueeze(-1).repeat(1, 1, 1, v.shape[-1])
    beta.requires_grad_(True)
    o, _ = fused_recurrent_linear_attn_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone())
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None
    print("test_beta_scalar_vector_equivalence:")
    print((o - o2).abs().max())
    # assert (o- o2).abs().max() < 1e-5
    print((q_grad - q_grad2).abs().max())
    print((k_grad - k_grad2).abs().max())
    print((v_grad - v_grad2).abs().max())
    print((beta_grad.sum(dim=-1) - beta_grad2).abs().max())


def test_chunk_fused_equivalence(q, k, v, beta, do):
    o2, _ = fused_chunk_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone(), 16)
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None

    o, _ = chunk_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone(), 32)
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None
    print("test_chunk_fused_equivalence:")
    print((o - o2).abs().max())
    # assert (o- o2).abs().max() < 1e-5
    print((q_grad - q_grad2).abs().max())
    print((k_grad - k_grad2).abs().max())
    print((v_grad - v_grad2).abs().max())
    print((beta_grad - beta_grad2).abs().max())


def test_chunk_fused_runtime(q, k, v, beta, do):
    print("test_chunk_fused_runtime:")
    print("Start warmup")
    for _ in range(30):
        o2, _ = fused_chunk_delta_rule(q, k, v, beta, 16)
        o2.backward(do, retain_graph=True)
        o, _ = chunk_delta_rule(q, k, v, beta, 32)
        o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    print("Warmup Done")

    start = time.time()
    for _ in range(100):
        o2, _ = fused_chunk_delta_rule(q, k, v, beta, 16)
        o2.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    for _ in range(100):
        o2, _ = chunk_delta_rule(q, k, v, beta, 32)
        o2.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    print(time.time() - start)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    q, k, v, beta, do = gen_inputs()

    test_beta_scalar_vector_equivalence(q, k, v, beta, do)
    # breakpoint()
    test_chunk_fused_equivalence(q, k, v, beta, do)
    # breakpoint()
    test_chunk_fused_runtime(q, k, v, beta, do)

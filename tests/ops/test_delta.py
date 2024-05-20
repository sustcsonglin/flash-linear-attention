from fla.ops.delta_rule import fused_chunk_delta_rule, chunk_delta_rule, fused_recurrent_linear_attn_delta_rule
import torch
import time

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    seq_len = 2048
    b = 8
    h = 8
    d = 256
    k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, d), p=2, dim=-1)
    v = torch.randn(b, h, seq_len, d)  
    q = torch.randn(b, h, seq_len, d)
    beta = torch.rand(b, h, seq_len).sigmoid()
    require_grad = True
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
    print((o- o2).abs().max())
    # assert (o- o2).abs().max() < 1e-5
    print((q_grad - q_grad2).abs().max())
    print((k_grad - k_grad2).abs().max())
    print((v_grad - v_grad2).abs().max())
    print((beta_grad - beta_grad2).abs().max())
    # breakpoint()

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




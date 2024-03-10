# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import re

import torch


def delta_rule_recurrence(q, k, v, beta):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o

    
def delta_rule_backward(q, k, v, do):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * d_k ** -0.5
    v_modified = torch.empty_like(v)

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        v_modified[:, :, i] = _v
        S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)

    b, h, l, d_k = q.shape
    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)
    DS = torch.zeros_like(S)


    for i in range(l-1, -1, -1):
        do_i = do[:, :, i]
        k_i = k[:, :, i]
        q_i = q[:, :, i]
        v_i = v_modified[:, :, i]

        DS_i = q_i.unsqueeze(-1) * do_i.unsqueeze(-2)

        DS += DS_i
        dv_i = torch.einsum('bhnm, bhn -> bhm', DS, k_i)
        d_v[:, :, i] = dv_i

        dk_i = torch.einsum('bhnm, bhm -> bhn', DS, v_i)
        d_k[:, :, i] = dk_i
        DS -= dv_i.unsqueeze(-2) * k_i.unsqueeze(-1)

    S = torch.zeros_like(DS)
    for i in range(l):
        _k = k[:, :, i]
        _do = do[:, :, i]
        _v = v_modified[:, :, i]
        S = S + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        d_q[:, :, i] = torch.einsum('bhm,bhdm->bhd', _do, S) * (q.shape[-1] ** -0.5)
        if i < l-1:
            _dv = d_v[:, :, i+1]
            d_k[:,:,i+1] -= torch.einsum('bhm,bhdm->bhd', _dv, S)
    return d_q, d_k, d_v        
        

if __name__ == '__main__':
    print("sss")
    B = 2
    H = 4
    L = 128
    DK = 64
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True) 
    k = (torch.randn(B, H, L, DK) / 10).cuda().requires_grad_(True) 
    v = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True) 
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    
    o = delta_rule_recurrence(q, k, v, beta).cuda()
    do = torch.randn(B, H, L, DK).cuda()
    o.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    from fla.ops.delta_rule.recurrent_fuse import \
        fused_recurrent_linear_attn_delta_rule as triton_delta_rule
    o2 = triton_delta_rule(q, k, v, beta)
    o2.backward(do)
    assert torch.allclose(o, o2, atol=1e-4), breakpoint()
    assert torch.allclose(q.grad, q_grad, atol=1e-4), breakpoint()
    assert torch.allclose(k.grad, k_grad, atol=1e-4), breakpoint()
    assert torch.allclose(v.grad, v_grad, atol=1e-4), breakpoint()
    assert torch.allclose(beta.grad, beta_grad, atol=1e-4), breakpoint()

    print("All pass.")


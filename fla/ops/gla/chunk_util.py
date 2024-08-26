import triton
import triton.language as tl


@triton.jit
def fwd_decay_cumsum(
    g,
    g_o,
    s_qk_h,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_g = g + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < K

    for i in range(BT):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += K
        p_go += K


@triton.jit
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    s_qk_h,
    scale,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):

    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < K

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * K + i_k * BK + tl.arange(0, BK))

    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.exp(_g) * scale
        _k *= tl.exp(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += K
        p_g += K
        p_k += K
        p_kg += K
        p_qg += K


@triton.jit
def bwd_decay_global_cumsum(
    dq_inner,
    dq_inter,
    dk_inner,
    dk_inter,
    q, k, g, dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < K
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT-1, -1, -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.exp(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.exp(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= K
        p_k -= K
        p_q -= K
        p_dq_inner -= K
        p_dk_inner -= K
        p_dq_inter -= K
        p_dk_inter -= K
        p_dg -= K

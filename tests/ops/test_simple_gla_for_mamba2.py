"""
Map Mamba-2's `mamba_chunk_scan_combined` kernel to FLA's `simple_gla` kernel

Dependencies:
$ pip install mamba-ssm==2.2.2 triton==2.3.1

Reference: `ssd_minimal_discrete` and `test_correctness` in mamba repository:
https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/modules/ssd_minimal.py#L82
"""

# Disable the prints:
# FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pytest
import torch

from fla.ops.simple_gla.naive import torch_simple_gla, torch_simple_gla_recurrent
from fla.ops.simple_gla import chunk_simple_gla

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete

dtype_mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@pytest.mark.parametrize("vary_A", [True, False])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "bfloat16"])
def test_gla_to_mamba2(vary_A, dtype_str):
    torch.manual_seed(42)

    # Dimensions, Denoted (B, T, Q, D, P) in Mamba2 paper
    batch, seqlen, chunk_size, dim, headdim = 2, 512, 8, 64, 16
    nheads = dim // headdim  # (H) in the paper
    ngroups = nheads # (G) in the paper; NOTE: do not use group-query here
    dstate = 64  # (N) in the paper
    device = "cuda"
    dtype = dtype_mapping[dtype_str]
    atol = 5e-4 if dtype_str == "float32" else 1e-2

    x = 0.1 * torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = torch.ones(batch, seqlen, nheads, dtype=dtype, device=device)  # dt=1 can be ignored

    if vary_A:
        A = -0.1 * torch.rand(1, seqlen, nheads, dtype=dtype, device=device)
    else:  # constant A for all position
        A = -0.1 * torch.rand(nheads, dtype=dtype, device=device)

    B = 0.1 * torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = 0.1 * torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)

    y_ssd, final_ssd = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)

    if not vary_A:
        # NOTE: fused kernel does not support varying A with time
        y_fuse, final_fuse = mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size, D=None, return_final_states=True
        )
        assert y_ssd.allclose(y_fuse, 0, atol), (
            f"y diff: {torch.abs(y_ssd - y_fuse).max()}"
        )
        # fused kernel upcasts state to float32
        # https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/triton/ssd_combined.py#L650
        final_fuse = final_fuse.to(dtype)
        assert final_ssd.allclose(final_fuse, 0, atol), (
            f"final diff: {torch.abs(final_ssd - final_fuse).max()}"
        )

    # mapping inputs Mamba2 -> FLA
    # C, B, X: [batch, seq, head, hidden] -> [batch, head, seq, hidden]
    # g: [batch, seq, head] -> [batch, head, seq]
    q = C.transpose(1, 2)
    k = B.transpose(1, 2)
    v = x.transpose(1, 2)
    g = (A*dt).transpose(1, 2)

    # mapping outputs Mamba2 -> FLA
    y_rearrange = y_ssd.transpose(1, 2)
    final_rearrange = final_ssd.transpose(2, 3)

    # comparing output results between FLA kernel and Mamba2 kernel
    outputs_gla_recurrent = torch_simple_gla_recurrent(
        q, k, v, g, scale=1.0
    )
    assert y_rearrange.allclose(outputs_gla_recurrent, 0, atol), (
        f"y diff: {torch.abs(y_rearrange - outputs_gla_recurrent).max()}"
    )

    outputs_gla_simple = torch_simple_gla(
        q, k, v, g, scale=1.0
    )
    assert y_rearrange.allclose(outputs_gla_simple, 0, atol), (
        f"y diff: {torch.abs(y_rearrange - outputs_gla_simple).max()}"
    )

    outputs_gla_fuse ,final_gla_fuse = chunk_simple_gla(
        q, k, v, g, scale=1.0, output_final_state=True
    )
    assert y_rearrange.allclose(outputs_gla_fuse, 0, atol), (
        f"y diff: {torch.abs(y_rearrange - outputs_gla_fuse).max()}"
    )
    final_gla_fuse = final_gla_fuse.to(dtype)  # states hard-coded to float32 in FLA kernel
    assert final_rearrange.allclose(final_gla_fuse, 0, atol), (
        f"final diff: {torch.abs(final_ssd - final_gla_fuse).max()}"
    )

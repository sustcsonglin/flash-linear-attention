# -*- coding: utf-8 -*-

import pytest
import torch

from fla.modules.convolution import ShortConvolution


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [100, 500, 1])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("kernel_size", [4])
def test_shortconv(batch_size: int, seq_len: int, hidden_size: int, kernel_size: int):
    naive_conv = ShortConvolution(hidden_size, kernel_size, activation='silu', use_causal_conv=False).cuda()
    causal_conv = ShortConvolution(hidden_size, kernel_size, activation='silu').cuda()
    causal_conv.weight.data.copy_(naive_conv.weight.data)
    if causal_conv.bias is not None:
        causal_conv.bias.data.copy_(naive_conv.bias.data)

    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    y_naive = naive_conv(x)
    y_causal = causal_conv(x)
    assert y_naive.shape == x.shape
    assert y_causal.shape == x.shape
    assert torch.allclose(y_naive, y_causal), f"{y_naive}\n{y_causal}"


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [10])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("kernel_size", [4])
def test_shortconv_cache(batch_size: int, seq_len: int, hidden_size: int, kernel_size: int):
    naive_conv = ShortConvolution(hidden_size, kernel_size, use_causal_conv=False).cuda()
    causal_conv = ShortConvolution(hidden_size, kernel_size).cuda()
    causal_conv.weight.data.copy_(naive_conv.weight.data)
    if causal_conv.bias is not None:
        causal_conv.bias.data.copy_(naive_conv.bias.data)

    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    y = naive_conv(x)
    cache_naive = x.new_zeros(batch_size, hidden_size, kernel_size)
    cache_causal = x.new_zeros(batch_size, hidden_size, kernel_size)
    for i in range(seq_len):
        y_naive, cache_naive = naive_conv(x[:, i:i+1], cache_naive)
        y_causal, cache_causal = causal_conv(x[:, i:i+1], cache_causal)
        assert torch.allclose(cache_naive, cache_causal), f"Step {i}\n{cache_naive}\n{cache_causal}"
        assert torch.allclose(y_naive, y[:, i]), f"Step {i}\n{y[:, i]}\n{naive_conv}:\n{y_naive}\n{causal_conv}:\n{y_causal}"
        assert torch.allclose(y_causal, y[:, i]), f"Step {i}\n{y[:, i]}\n{naive_conv}:\n{y_naive}\n{causal_conv}:\n{y_causal}"

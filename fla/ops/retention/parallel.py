# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang
import torch
from typing import Optional, Tuple
from fla.ops.simple_gla.parallel import parallel_simple_gla

def parallel_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    output_attentions: bool = False,
    head_first: bool = True,
    offsets: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.
        offsets (Optional[torch.LongTensor]):
            Offsets of shape `[N+1]` defining the bos/eos positions of `N` variable-length sequences in the batch.
            For example,
            if `offsets` is `[0, 1, 3, 6, 10, 15]`, there are `N=5` sequences with lengths 1, 2, 3, 4 and 5 respectively.
            If provided, the inputs are concatenated and the batch size `B` is expected to be 1.
            Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if head_first:
        n_heads = q.shape[1]
    else:
        n_heads = q.shape[2]
    s = (1 - q.new_tensor(2., dtype=torch.float, device=q.device).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float, device=q.device))).log()
    if head_first:
        g = s[None, :, None].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    else:
        g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()

    return parallel_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g=g,
        output_attentions=output_attentions,
        head_first=head_first,
        offsets=offsets
    )

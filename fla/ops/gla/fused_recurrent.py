# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch

from fla.ops.common.fused_recurrent import fused_recurrent


def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        gk (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        gv (torch.Tensor):
            Forget gates of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]` applied to values.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = fused_recurrent(q, k, v, None, gk, gv, scale, initial_state, output_final_state, reverse, head_first)
    return o, final_state

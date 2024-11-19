# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch

from fla.ops.common.fused_recurrent import fused_recurrent


def fused_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v, g = map(lambda x: x.transpose(1, 2), (q, k, v, g))
    o, final_state = fused_recurrent(q, k, v, g, None, None, scale, initial_state, output_final_state, reverse)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state

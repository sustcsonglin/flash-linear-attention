# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch

from fla.ops.common.fused_recurrent import fused_recurrent


def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor = None,
    gv: torch.Tensor = None,
    scale: int = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = fused_recurrent(q, k, v, None, gk, gv, scale, initial_state, output_final_state, reverse)
    return o, final_state

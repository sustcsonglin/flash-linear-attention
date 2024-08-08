# -*- coding: utf-8 -*-

from .recurrent_fuse import fused_recurrent_linear_attn_delta_rule
from .recurrent_fuse_betavec import fused_recurrent_linear_attn_delta_rule_betavec

# from .chunk_fn import chunk_linear_attn_delta_rule
from .chunk_fused import fused_chunk_delta_rule

__all__ = [
    'fused_recurrent_linear_attn_delta_rule',
    'fused_chunk_delta_rule',
    'fused_recurrent_linear_attn_delta_rule_betavec'
]


# -*- coding: utf-8 -*-

from .recurrent_fuse import fused_recurrent_linear_attn_delta_rule
from .chunk_fn import chunk_linear_attn_delta_rule

__all__ = [
    'fused_recurrent_linear_attn_delta_rule',
    'chunk_linear_attn_delta_rule'
]


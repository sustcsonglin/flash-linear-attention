# -*- coding: utf-8 -*-

# from .chunk import chunk_linear_attn
# from .chunk_fuse import fused_chunk_linear_attn
from .recurrent_fuse import fused_recurrent_linear_attn_delta_rule

__all__ = [
    'fused_recurrent_linear_attn_delta_rule',
]


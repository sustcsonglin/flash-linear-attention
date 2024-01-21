# -*- coding: utf-8 -*-

from fla.ops.torch import (naive_chunk_based, naive_parallel_based,
                           naive_recurrent_gla, naive_retention)
from fla.ops.triton import (chunk_gla, chunk_retention, fused_chunk_based,
                            fused_chunk_gla, fused_chunk_retention,
                            fused_recurrent_gla, fused_recurrent_retention,
                            parallel_based, parallel_retention)

__all__ = [
    'naive_chunk_based',
    'naive_parallel_based',
    'naive_recurrent_gla',
    'naive_retention',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention',
    'fused_recurrent_gla',
    'fused_recurrent_retention',
    'parallel_based',
    'parallel_retention',
]

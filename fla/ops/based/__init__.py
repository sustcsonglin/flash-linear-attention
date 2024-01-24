# -*- coding: utf-8 -*-

from .chunk_fuse import fused_chunk_based
from .naive import naive_chunk_based, naive_parallel_based
from .parallel import parallel_based

__all__ = [
    'fused_chunk_based',
    'parallel_based',
    'naive_chunk_based',
    'naive_parallel_based'
]

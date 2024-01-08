# -*- coding: utf-8 -*-

from .chunk_fuse import fused_chunk_retention
from .naive import naive_retention
from .parallel import parallel_retention
from .recurrent_fuse import fused_recurrent_retention
from .chunk import chunk_retention

__all__ = ['fused_chunk_retention', 'naive_retention', 'parallel_retention',
           'fused_recurrent_retention', 'chunk_retention']
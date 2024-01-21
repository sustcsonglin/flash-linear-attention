# -*- coding: utf-8 -*-

from .chunk_fuse import fused_chunk_based
from .parallel import parallel_based

__all__ = ["parallel_based", "fused_chunk_based"]

# -*- coding: utf-8 -*-

from fla.layers import (BasedLinearAttention, GatedLinearAttention,
                        MultiScaleRetention)
from fla.models import GLAForCausalLM, GLAModel, RetNetForCausalLM, RetNetModel
from fla.ops import (chunk_gla, chunk_retention, fused_chunk_based,
                     fused_chunk_gla, fused_chunk_retention)

__all__ = [
    'BasedLinearAttention',
    'GatedLinearAttention',
    'MultiScaleRetention',
    'GLAForCausalLM',
    'GLAModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention'
]

__version__ = '0.1'

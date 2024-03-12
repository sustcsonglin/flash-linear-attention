# -*- coding: utf-8 -*-

from fla.layers import (ABCAttention, BasedLinearAttention, DeltaNet,
                        GatedLinearAttention, LinearAttention,
                        MultiScaleRetention, ReBasedLinearAttention)
from fla.models import (ABCForCausalLM, ABCModel, DeltaNetForCausalLM,
                        DeltaNetModel, GLAForCausalLM, GLAModel,
                        LinearAttentionForCausalLM, LinearAttentionModel,
                        RetNetForCausalLM, RetNetModel)
from fla.ops import (chunk_gla, chunk_retention, fused_chunk_based,
                     fused_chunk_gla, fused_chunk_retention)

__all__ = [
    'ABCAttention',
    'BasedLinearAttention',
    'DeltaNet',
    'GatedLinearAttention',
    'LinearAttention',
    'MultiScaleRetention',
    'ReBasedLinearAttention',
    'ABCForCausalLM',
    'ABCModel',
    'DeltaNetForCausalLM',
    'DeltaNetModel',
    'GLAForCausalLM',
    'GLAModel',
    'LinearAttentionForCausalLM',
    'LinearAttentionModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention'
]

__version__ = '0.1'

# -*- coding: utf-8 -*-

from fla.layers import (ABCAttention, BasedLinearAttention, DeltaNet,
                        GatedLinearAttention, HGRN2Attention, LinearAttention,
                        MultiScaleRetention, ReBasedLinearAttention)
from fla.models import (ABCForCausalLM, ABCModel, DeltaNetForCausalLM,
                        DeltaNetModel, GLAForCausalLM, GLAModel,
                        HGRN2ForCausalLM, HGRN2Model,
                        LinearAttentionForCausalLM, LinearAttentionModel,
                        RetNetForCausalLM, RetNetModel, TransformerForCausalLM,
                        TransformerModel)
from fla.ops import (chunk_gla, chunk_retention, fused_chunk_based,
                     fused_chunk_gla, fused_chunk_retention)

__all__ = [
    'ABCAttention',
    'BasedLinearAttention',
    'DeltaNet',
    'HGRN2Attention',
    'GatedLinearAttention',
    'LinearAttention',
    'MultiScaleRetention',
    'ReBasedLinearAttention',
    'ABCForCausalLM',
    'ABCModel',
    'DeltaNetForCausalLM',
    'DeltaNetModel',
    'HGRN2ForCausalLM',
    'HGRN2Model',
    'GLAForCausalLM',
    'GLAModel',
    'LinearAttentionForCausalLM',
    'LinearAttentionModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'TransformerForCausalLM',
    'TransformerModel',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention'
]

__version__ = '0.1'

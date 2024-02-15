# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel

from fla.models.gla.configuration_gla import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM

AutoConfig.register(GLAConfig.model_type, GLAConfig)
AutoModel.register(GLAConfig, GLAForCausalLM)

__all__ = ['GLAConfig', 'GLAForCausalLM']

# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.bitnet.configuration_bitnet import BitNetConfig
from fla.models.bitnet.modeling_bitnet import BitNetForCausalLM, BitNetModel

AutoConfig.register(BitNetConfig.model_type, BitNetConfig)
AutoModel.register(BitNetConfig, BitNetModel)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM)


__all__ = ['BitNetConfig', 'BitNetForCausalLM', 'BitNetModel']

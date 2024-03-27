# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba.configuration_mamba import MambaConfig
from fla.models.mamba.modeling_mamba import MambaForCausalLM, MambaModel, MambaBlock

AutoConfig.register(MambaConfig.model_type, MambaConfig)
AutoModel.register(MambaConfig, MambaModel)
AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel', 'MambaBlock']

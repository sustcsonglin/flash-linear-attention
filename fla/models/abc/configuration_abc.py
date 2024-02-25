# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class ABCConfig(PretrainedConfig):

    model_type = 'abc'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        gate_low_rank_dim: int = 16,
        clamp_min: float = -32,
        clamp_max: float = 32,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_slots: Optional[int] = None,
        exapnd_k: float = 1,
        exapnd_v: float = 1,
        attention_dropout: float = 0.,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.gate_low_rank_dim = gate_low_rank_dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_slots = num_slots
        self.expand_k = exapnd_k
        self.expand_v = exapnd_v
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

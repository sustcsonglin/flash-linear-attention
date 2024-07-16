# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class GSAConfig(PretrainedConfig):

    model_type = 'gsa'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        gate_low_rank_dim: Optional[int] = None,
        gate_logit_normalizer: Optional[int] = 16,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        num_slots: Optional[int] = 64,
        use_short_conv: bool = False,
        conv_size: int = 4,
        exapnd_k: float = 1,
        exapnd_v: float = 1,
        feature_map: str = 'swish',
        use_rope: bool = False,
        use_output_gate: bool = False,
        use_norm: bool = True,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: Optional[bool] = True,
        norm_first: bool = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
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
        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_slots = num_slots
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.expand_k = exapnd_k
        self.expand_v = exapnd_v
        self.feature_map = feature_map
        self.use_rope = use_rope
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_first = norm_first
        self.norm_eps = norm_eps
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

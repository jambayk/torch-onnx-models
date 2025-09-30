from __future__ import annotations

import dataclasses

import torch


# https://github.com/huggingface/transformers/blob/3e975acc8bf6d029ec0a54b1c5d0691489dfb051/src/transformers/models/auto/configuration_auto.py#L36
SUPPORTED_ARCHITECTURES = {
    # "ernie4_5",
    # "gemma",
    # "gemma2",
    # "gemma3",
    # "gptoss",
    # "granite",
    "llama",
    # "mistral",
    # "nemotron",
    # "olmo",
    # "phi",
    # "phi3",
    # "phi3small",
    # "phi3v",
    # "phi4mm",
    # "phimoe",
    # "qwen2",
    # "qwen3",
    # "smollm3",
}


DEFAULT_INT = -42


# self.vocab_size = vocab_size
# self.max_position_embeddings = max_position_embeddings
# self.hidden_size = hidden_size
# self.intermediate_size = intermediate_size
# self.num_hidden_layers = num_hidden_layers
# self.num_attention_heads = num_attention_heads

# # for backward compatibility
# if num_key_value_heads is None:
#     num_key_value_heads = num_attention_heads

# self.num_key_value_heads = num_key_value_heads
# self.hidden_act = hidden_act
# self.initializer_range = initializer_range
# self.rms_norm_eps = rms_norm_eps
# self.pretraining_tp = pretraining_tp
# self.use_cache = use_cache
# self.rope_theta = rope_theta
# self.rope_scaling = rope_scaling
# self.attention_bias = attention_bias
# self.attention_dropout = attention_dropout
# self.mlp_bias = mlp_bias
# self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
# # Validate the correctness of rotary position embeddings parameters
# # BC: if there is a 'type' field, copy it it to 'rope_type'.
# if self.rope_scaling is not None and "type" in self.rope_scaling:
#     self.rope_scaling["rope_type"] = self.rope_scaling["type"]
# rope_config_validation(self)

@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    head_dim: int = DEFAULT_INT
    num_attention_heads: int = DEFAULT_INT
    num_key_value_heads: int = DEFAULT_INT
    num_hidden_layers: int = DEFAULT_INT
    vocab_size: int = DEFAULT_INT
    hidden_size: int = DEFAULT_INT
    intermediate_size: int = DEFAULT_INT
    hidden_act: str | None = None
    pad_token_id: int = DEFAULT_INT
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False

    # Rotary embedding config
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    max_position_embeddings: int = DEFAULT_INT
    partial_rotary_factor: float = 1.0  # 1.0 means no partial RoPE

    @classmethod
    def from_transformers(cls, config) -> ArchitectureConfig:
        if config.model_type not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Model type '{config.model_type}' not supported. Supported architectures: {SUPPORTED_ARCHITECTURES}"
            )

        options = dict(
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=(
                getattr(config, "num_key_value_heads", config.num_attention_heads)
            ),
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=(
                getattr(config, "intermediate_size", 4 * config.hidden_size)
            ),
            hidden_act=(getattr(config, "hidden_act", None)),
            pad_token_id=(getattr(config, "pad_token_id", 0)),  # FIXME
            rms_norm_eps=(getattr(config, "rms_norm_eps", 1e-6)),
            attention_bias=(getattr(config, "add_bias_kv", False)),
            mlp_bias=(getattr(config, "use_mlp_bias", False)),
            rope_type="default",  # only support default for now
            rope_theta=(getattr(config, "rope_theta", 10_000.0)),
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=(getattr(config, "partial_rotary_factor", 1.0)),
        )

        return cls(**options)


@dataclasses.dataclass
class ExportConfig:
    nothing_yet: bool = True

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


@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    head_dim: int = -42
    num_attention_heads: int = -42
    num_key_value_heads: int = -42
    num_hidden_layers: int = -42
    vocab_size: int = -42
    hidden_size: int = -42
    intermediate_size: int = -42
    hidden_act: str | None = None
    pad_token_id: int = -42
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False

    # Rotary embedding config
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    max_position_embeddings: int = -42
    partial_rotary_factor: float = 1.0  # 1.0 means no partial RoPE

    @classmethod
    def from_transformers(cls, config) -> ArchitectureConfig:
        if getattr(config, "model_type") != "llama":
            raise ValueError("Only llama model is supported yet")

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

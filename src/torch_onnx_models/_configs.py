from __future__ import annotations

import dataclasses

import torch

# https://github.com/huggingface/transformers/blob/3e975acc8bf6d029ec0a54b1c5d0691489dfb051/src/transformers/models/auto/configuration_auto.py#L36
SUPPORTED_ARCHITECTURES = {
    # "ernie4_5",
    # "gemma",
    # "gemma2",
    # "gemma3",
    "gemma3_text",
    # "gptoss",
    # "granite",
    "llama",
    "mistral",
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


@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    vocab_size: int = DEFAULT_INT
    max_position_embeddings: int = DEFAULT_INT
    hidden_size: int = DEFAULT_INT
    intermediate_size: int = DEFAULT_INT
    num_hidden_layers: int = DEFAULT_INT
    num_attention_heads: int = DEFAULT_INT
    num_key_value_heads: int = DEFAULT_INT
    hidden_act: str | None = None

    # attention config
    use_qk_norm: bool = False  # only for Gemma-3
    layer_types: list[str] | None = None  # only for Gemma-3
    sliding_window: int | None = None  # only for Gemma-3

    rms_norm_eps: float = 1e-6
    rms_norm_all_fp32: bool = False  # only for Gemma-3
    rms_norm_offset: float | None = None  # only for Gemma-3

    # Rotary embedding config
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0  # 1.0 means no partial RoPE
    rope_local_base_freq: float | None = None  # only for Gemma-3

    attention_bias: bool = False
    mlp_bias: bool = False

    head_dim: int = DEFAULT_INT

    pad_token_id: int = DEFAULT_INT

    @classmethod
    def from_transformers(cls, config) -> ArchitectureConfig:
        if config.model_type not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Model type '{config.model_type}' not supported. Supported architectures: {SUPPORTED_ARCHITECTURES}"
            )

        options = dict(
            head_dim=(
                config.head_dim
                if (hasattr(config, "head_dim") and config.head_dim is not None)
                else config.hidden_size // config.num_attention_heads
            ),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=(getattr(config, "num_key_value_heads", config.num_attention_heads)),
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=(getattr(config, "intermediate_size", 4 * config.hidden_size)),
            hidden_act=(getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None)),
            use_qk_norm=(config.model_type == "gemma3_text"),
            # consider using the layer types directly from the hf config
            layer_types=(
                [
                    "local" if (i + 1) % config.sliding_window_pattern != 0 else "global"
                    for i in range(config.num_hidden_layers)
                ]
                if hasattr(config, "sliding_window_pattern")
                else None
            ),
            sliding_window=(getattr(config, "sliding_window", None)),
            pad_token_id=(getattr(config, "pad_token_id", 0)),  # FIXME
            rms_norm_eps=(getattr(config, "rms_norm_eps", 1e-6)),
            rms_norm_all_fp32=(config.model_type == "gemma3_text"),
            rms_norm_offset=(1.0 if config.model_type == "gemma3_text" else None),
            attention_bias=(getattr(config, "add_bias_kv", False)),
            mlp_bias=(getattr(config, "use_mlp_bias", False)),
            # how much older transformers versions are we supporting?
            rope_type=(
                config.rope_scaling.get("rope_type")
                if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict)
                else "default"
            ),
            rope_theta=(getattr(config, "rope_theta", 10_000.0)),
            rope_scaling=(getattr(config, "rope_scaling", None)),
            partial_rotary_factor=(getattr(config, "partial_rotary_factor", 1.0)),
            rope_local_base_freq=(getattr(config, "rope_local_base_freq", None)),
            max_position_embeddings=config.max_position_embeddings,
        )

        return cls(**options)


@dataclasses.dataclass
class ExportConfig:
    nothing_yet: bool = True

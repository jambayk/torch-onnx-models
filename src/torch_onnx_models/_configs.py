from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ArchitectureConfig:
    dtype: torch.dtype = torch.float32

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
    def from_transformers(cls, config: dict) -> ArchitectureConfig:
        if config.get("model_type") != "llama":
            raise ValueError("Only llama model is supported yet")
        return cls(
            head_dim=config["hidden_size"] // config["num_attention_heads"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=(
                config.get("num_key_value_heads", config["num_attention_heads"])
            ),
            num_hidden_layers=config["num_hidden_layers"],
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=(
                config.get("intermediate_size", 4 * config["hidden_size"])
            ),
            hidden_act=(config.get("hidden_act", None)),
            pad_token_id=(config.get("pad_token_id", 0)),  # FIXME
            rms_norm_eps=(config.get("rms_norm_eps", 1e-6)),
            attention_bias=(config.get("add_bias_kv", False)),
            mlp_bias=(config.get("use_mlp_bias", False)),
            rope_type="default",  # only support default for now
            rope_theta=(config.get("rope_theta", 10_000.0)),
            max_position_embeddings=config["max_position_embeddings"],
            partial_rotary_factor=(config.get("partial_rotary_factor", 1.0)),
            dtype=torch.float16,  # TODO: Fix this
        )


@dataclasses.dataclass
class ExportConfig:
    nothing_yet: bool = True

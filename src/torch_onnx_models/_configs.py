from __future__ import annotations

import dataclasses

import torch


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


@dataclasses.dataclass
class ExportConfig:
    dtype: torch.dtype = torch.float32

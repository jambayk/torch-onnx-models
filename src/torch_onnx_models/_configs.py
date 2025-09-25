from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    head_dim: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    num_hidden_layers: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    hidden_act: str | None = None
    pad_token_id: int | None = None
    rms_norm_eps: float | None = None
    attention_bias: bool = False
    mlp_bias: bool = False

    # Rotary embedding config
    rope_type: str | None = None
    rope_theta: float = 10_000.0
    max_position_embeddings: int | None = None


@dataclasses.dataclass
class ExportConfig:
    dtype: torch.dtype = torch.float32

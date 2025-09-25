from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    pad_token_id: int
    rms_norm_eps: float
    attention_bias: bool  # = False
    mlp_bias: bool  # = False

    # Rotary embedding config
    rope_type: str | None  # = None
    max_position_embeddings: int

    @classmethod
    def from_default(cls, **kwargs) -> ArchitectureConfig:
        defaults = {
            "attention_bias": False,
            "mlp_bias": False,
            "rope_type": None,
        }
        defaults.update(kwargs)
        return cls(**defaults)


@dataclasses.dataclass
class ExportConfig:
    dtype: torch.dtype = torch.float32

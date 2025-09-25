import dataclasses
import torch


@dataclasses.dataclass
class ArchitectureConfig:
    # Config from transformers
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    attention_bias: bool = False
    mlp_bias: bool = False


@dataclasses.dataclass
class ExportConfig:
    dtype: torch.dtype = torch.float32

import dataclasses
import torch


@dataclasses.dataclass
class ModelConfig:
    # Config from transformers
    pass


@dataclasses.dataclass
class ExportConfig:
    dtype: torch.dtype = torch.float32

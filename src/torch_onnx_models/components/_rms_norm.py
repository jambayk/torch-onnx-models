from __future__ import annotations

__all__ = ["RMSNorm"]

import torch
from torch import nn
from torch_onnx_models.components._rms_norm_utils import apply_rms_norm
from torch_onnx_models import _configs


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, config: _configs.ArchitectureConfig):
        super().__init__()
        # Mark: weights
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = config.rms_norm_eps
        self.all_fp32 = config.rms_norm_all_fp32
        self.offset = config.rms_norm_offset

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        weight = self.weight
        # it's okay to do these casts since redudant casts will be optimized away
        if self.all_fp32:
            hidden_states = hidden_states.float()
            weight = weight.float()
        if self.offset is not None:
            weight = weight + self.offset
        return apply_rms_norm(x=hidden_states, weight=weight, eps=self.variance_epsilon).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

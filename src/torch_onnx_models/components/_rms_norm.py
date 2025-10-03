from __future__ import annotations

__all__ = ["RMSNorm"]

import torch
from torch import nn
from torch_onnx_models.components._rms_norm_utils import apply_rms_norm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Mark: weights
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return apply_rms_norm(
            x=hidden_states, weight=self.weight, eps=self.variance_epsilon
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

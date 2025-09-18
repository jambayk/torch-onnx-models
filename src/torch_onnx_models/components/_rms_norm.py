from __future__ import annotations

import torch
from torch import nn
from torch_onnx_models.components._rms_norm_utils import (
    apply_rms_norm,
    apply_rms_norm_decomposed,
    apply_rms_norm_contrib,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # just for testing
        rms_norm_func = apply_rms_norm
        # rms_norm_func = apply_rms_norm_decomposed
        # rms_norm_func = apply_rms_norm_contrib
        print("Using rms norm func:", rms_norm_func.__name__)
        return rms_norm_func(x=hidden_states, weight=self.weight, eps=self.variance_epsilon)

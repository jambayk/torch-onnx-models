from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, *, hidden_size: int, eps: float = 1e-6, mode: str):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._config = mode

    def forward(self, hidden_states):
        if torch.onnx.is_in_onnx_export():
            if self._config == "ort":
                return torch.onnx.ops.symbolic(
                    "ai.onnx::SimplifiedLayerNormalization",
                    [hidden_states, self.weight],
                    attrs={"epsilon": self.variance_epsilon},
                    dtype=hidden_states.dtype,
                    shape=hidden_states.shape,
                    version=1,
                )
        # This will produce the correct ONNX standard ops based on the opset requested
        # rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
        return torch.ops.aten.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)

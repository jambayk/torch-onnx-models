from __future__ import annotations

import torch
from torch import nn
from torch_onnx_models import _config


class RMSNorm(nn.Module):
    def __init__(self, *, hidden_size: int, eps: float = 1e-6, config: _config.Config):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.variance_epsilon = eps
        self._config = config

    def forward(self, hidden_states):
        if torch.onnx.is_in_onnx_export():
            if self._config.opset == "ort":
                return torch.onnx.ops.symbolic(
                    "com.microsoft::SkipSimplifiedLayerNormalization",
                    [hidden_states, None, self.weight],
                    attrs={"epsilon": self.variance_epsilon},
                    dtype=hidden_states.dtype,
                    shape=hidden_states.shape,
                    version=1,
                )
            else:
                # This will produce the correct ONNX standard ops based on the opset requested
                return torch.ops.aten.rms_norm(
                    hidden_states,
                    (hidden_states.size(-1),),
                    self.weight,
                    self.variance_epsilon,
                )
        else:
            # rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
            return torch.ops.aten.rms_norm(
                hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon
            )

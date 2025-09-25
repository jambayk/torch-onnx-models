import torch
from torch import nn

from torch_onnx_models.components import _activations
from torch_onnx_models import _configs


class Phi3MLP(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = _activations.get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class LlamaMLP(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = _activations.get_activation(config.hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

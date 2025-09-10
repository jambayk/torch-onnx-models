import torch
from torch import nn

from torch_onnx_models.components import _activations


class Phi3MLP(nn.Module):
    # TODO: Pin down the config implementation
    def __init__(self, config):
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

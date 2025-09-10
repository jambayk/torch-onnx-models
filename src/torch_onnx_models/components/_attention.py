from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):

    # these init values will later come from the model config
    def __init__(self, *, hidden_size: int, head_dim: int, num_attention_heads: int, num_key_value_heads: int, merged_qkv: bool = False, attention_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # models like gemma have different scaling, generalize later
        self.scaling = head_dim**-0.5

        # make this configurable or not?
        self.merged_qkv = merged_qkv
        if self.merged_qkv:
            self.qkv_proj = nn.Linear(
                self.hidden_size, (self.num_attention_heads + 2* self.num_key_value_heads) * self.head_dim, bias=attention_bias
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_attention_heads * self.head_dim, bias=attention_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
            )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     causal_mask: torch.Tensor,
    #     position_ids: torch.Tensor)

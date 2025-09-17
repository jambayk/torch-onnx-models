from __future__ import annotations

import torch
from torch import nn
from argparse import Namespace
from torch_onnx_models.components._attention_utils import attention, attention_decomposed
from torch_onnx_models.components._rotary_embedding_utils import apply_rope, apply_rope_decomposed


class Attention(nn.Module):

    # replace config typing with actual config class later
    def __init__(self, config: Namespace):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # models like gemma have different scaling, generalize later
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # just for testing
        rope_func = apply_rope
        # rope_func = apply_rope_decomposed
        query_states = rope_func(
            x=query_states,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            position_ids=position_ids,
            num_heads=self.num_attention_heads,
        )
        key_states = rope_func(
            x=key_states,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            position_ids=position_ids,
            num_heads=self.num_key_value_heads,
        )

        attention_func = attention
        # attention_func = attention_decomposed
        attn_output, present_key, present_value = attention_func(
            query=query_states,
            key=key_states,
            value=value_states,
            bias=attention_bias,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key, present_value

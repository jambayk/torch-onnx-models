from __future__ import annotations

import torch
from torch import nn
from argparse import Namespace
from torch_onnx_models.components._attention_utils import attention, attention_contrib_mha
from torch_onnx_models.components._rotary_embedding_utils import apply_rope


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
        # print("query_states", query_states.shape)
        # print("key_states", key_states.shape)
        # print("value_states", value_states.shape)
        # print("attention_bias", attention_bias.shape)
        # print("position_ids", position_ids.shape)
        # print("cos_cache", cos_cache.shape)
        # print("sin_cache", sin_cache.shape)
        # print("past_key", None if past_key is None else past_key.shape)
        # print("past_value", None if past_value is None else past_value.shape)
        query_states = apply_rope(
            x=query_states,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            position_ids=position_ids,
            num_heads=self.num_attention_heads,
        )
        key_states = apply_rope(
            x=key_states,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            position_ids=position_ids,
            num_heads=self.num_key_value_heads,
        )

        attn_output, present_key, present_value = attention(
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

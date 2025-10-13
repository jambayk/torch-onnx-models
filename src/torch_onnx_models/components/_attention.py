from __future__ import annotations

import torch
from torch import nn

from torch_onnx_models.components._attention_utils import attention
from torch_onnx_models.components._rotary_embedding_utils import apply_rotary_pos_emb
from torch_onnx_models.components._rms_norm import RMSNorm
from torch_onnx_models import _configs


class Attention(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # models like gemma have different scaling, generalize later
        self.scaling = self.head_dim**-0.5
        self.rotary_embedding_dim = (
            0 if config.partial_rotary_factor == 1.0 else int(self.head_dim * config.partial_rotary_factor)
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = apply_rotary_pos_emb(
            x=query_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
        )
        key_states = apply_rotary_pos_emb(
            x=key_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_key_value_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
        )

        attn_output, present_key, present_value = attention(
            query=query_states,
            key=key_states,
            value=value_states,
            bias=attention_bias,
            past_key=past_key_value[0] if past_key_value is not None else None,
            past_value=past_key_value[1] if past_key_value is not None else None,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, (present_key, present_value)

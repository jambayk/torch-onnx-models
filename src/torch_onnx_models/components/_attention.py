from __future__ import annotations

import torch
from torch import nn
from argparse import Namespace
from torch_onnx_models import _global_configs


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
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        rope_cos_freqs: torch.Tensor,
        rope_sin_freqs: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # this needs to be extracted into a reusable function
        if torch.onnx.is_in_onnx_export() and _global_configs.target == "ort":
            # need to wrap as tensor since torch.onnx.ops.symbolic doesn't support symbolic int
            total_seq_length = torch.tensor(attention_mask.size(1), dtype=torch.int32)
            past_seq_length = (attention_mask.sum(1, keepdim=True) - 1).to(torch.int32)
            present_shape = (query_states.size(0), self.num_key_value_heads, attention_mask.size(1), self.head_dim)
            attention_output, present_key, present_value = torch.onnx.ops.symbolic_multi_out(
                "com.microsoft::GroupQueryAttention",
                [
                    query_states,
                    key_states,
                    value_states,
                    past_key,
                    past_value,
                    past_seq_length,
                    total_seq_length,
                    rope_cos_freqs,
                    rope_sin_freqs,
                ],
                attrs={
                    "do_rotary": 1,
                    "kv_num_heads": self.num_key_value_heads,
                    "num_heads": self.num_attention_heads,
                    "rotary_interleaved": 0,
                    "scaling": self.scaling,
                },
                dtypes=(hidden_states.dtype, hidden_states.dtype, hidden_states.dtype),
                shapes=(query_states.shape, present_shape, present_shape),
                version=1,
            )
        else:
            # get it from position ids so that the subgraph is shared across layers
            query_length = position_ids.shape[1]
            # don't need -1 since it's all relative
            all_indices = attention_mask.cumsum(-1)
            kv_indices = all_indices[:, None, :]
            q_indices = all_indices[:, -query_length:, None]
            full_mask = q_indices >= kv_indices
            full_mask &= attention_mask[:, None, :].to(torch.bool)
            # make the negative value configurable
            full_mask = torch.where(full_mask, 0.0, -10000.0)[:, None, :, :]

            # # Also need to make this into a function that chooses between the contrib op and torch.onnx op
            query_states = torch.onnx.ops.rotary_embedding(
                query_states,
                rope_cos_freqs,
                rope_sin_freqs,
                position_ids,
                interleaved=False,
                num_heads=self.num_attention_heads,
            )
            key_states = torch.onnx.ops.rotary_embedding(
                key_states,
                rope_cos_freqs,
                rope_sin_freqs,
                position_ids,
                interleaved=False,
                num_heads=self.num_key_value_heads,
            )

            # rotary embedding is not respective the shape specs. Do this for now to make it back to 3d
            query_states = query_states.view(*hidden_states.shape[:-1], -1)
            key_states = key_states.view(*hidden_states.shape[:-1], -1)

            attention_output, present_key, present_value, _ = torch.onnx.ops.attention(
                query_states,
                key_states,
                value_states,
                full_mask,
                past_key,
                past_value,
                kv_num_heads=self.num_key_value_heads,
                q_num_heads=self.num_attention_heads,
                scale=self.scaling,
            )

        attention_output = self.o_proj(attention_output)
        return attention_output, present_key, present_value

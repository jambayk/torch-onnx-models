from __future__ import annotations

import copy

import torch
from torch import nn

from torch_onnx_models.components import apply_rms_norm, create_attention_bias, initialize_rope, Attention, MLP, RMSNorm
from torch_onnx_models._configs import ArchitectureConfig


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, config)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class Gemma3TextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # original model is in bf16 and the modeling code has this in bf16
        # huggingface chose to keep the same modeling code and use the weight dtype
        # but this would lead to different results if the weight is float32
        # we chose to use a constant in bf16 to match the original model
        # https://github.com/huggingface/transformers/pull/29402
        # https://github.com/huggingface/transformers/issues/38702
        self.embed_scale = torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16).item()

        self.layers = nn.ModuleList([Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window

        self.norm = RMSNorm(config.hidden_size, config)
        self.rotary_emb = initialize_rope(config)
        # make this better, even transformers does the same for now
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_type = "default"
        config.rope_scaling = None
        self.rotary_emb_local = initialize_rope(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # embed tokens and positions
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        position_embeddings_dict = {
            "global": self.rotary_emb(position_ids),
            "local": self.rotary_emb_local(position_ids),
        }

        # get the attention bias
        attention_bias_dict = {
            "global": create_attention_bias(
                attention_mask=attention_mask, query_length=input_ids.shape[-1], dtype=hidden_states.dtype
            ),
            "local": create_attention_bias(
                attention_mask=attention_mask,
                query_length=input_ids.shape[-1],
                dtype=hidden_states.dtype,
                sliding_window=self.sliding_window,
            ),
        }

        present_key_values = []
        for layer, layer_type, past_key_value in zip(
            self.layers, self.layer_types, past_key_values or [None] * len(self.layers)
        ):
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values


class Gemma3CausalLMModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = Gemma3TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        return logits, present_key_values

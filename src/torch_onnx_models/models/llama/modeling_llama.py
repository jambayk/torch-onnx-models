from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch._subclasses.fake_tensor
from torch import nn

from torch_onnx_models import ArchitectureConfig, components

logger = logging.getLogger(__name__)


class LlamaRMSNorm(components.RMSNorm):
    pass


class LlamaAttention(components.Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = components.LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor | None,
        position_ids: torch.LongTensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_ids=position_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            past_key=past_key,
            past_value=past_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            # The buffers need to be concrete tensors
            cos_cache, sin_cache = components.create_rope_caches(config)
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # The mask is made causal in the attention layer
        # TODO(justinchuby): But we may not want to make it causal for other models

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_bias=attention_mask,
                position_ids=position_ids,
                past_key=past_key_values[i][0],
                past_value=past_key_values[i][1],
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)

        return logits

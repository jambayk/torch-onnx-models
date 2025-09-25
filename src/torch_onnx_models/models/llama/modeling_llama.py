from __future__ import annotations

import logging
from typing import Sequence

import torch
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
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_bias=attention_mask,
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


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: ArchitectureConfig):
        super().__init__()

        self.rope_type = config.rope_type or "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

    def forward(self, x, position_ids):
        # FIXME: What should this look like?
        components.apply_rope(
            x=x, cos_cache=None, sin_cache=None, position_ids=position_ids
        )


class LlamaModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
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
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.config = config

    def forward(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor,
        past_keys: Sequence[torch.Tensor],
        past_values: Sequence[torch.Tensor],
        cache_position: torch.LongTensor,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # TODO: Implement create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_keys=past_keys,
            past_values=past_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            # TODO(justinchuby): Compute cos_cache, sin_cache from positions?
            # FIXME: How should position_embeddings be handled?
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key=past_keys[i],
                past_value=past_values[i],
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_keys: Sequence[torch.Tensor],
        past_values: Sequence[torch.Tensor],
        cache_position: torch.LongTensor,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_keys=past_keys,
            past_values=past_values,
            cache_position=cache_position,
        )

        logits = self.lm_head(hidden_states)

        return logits

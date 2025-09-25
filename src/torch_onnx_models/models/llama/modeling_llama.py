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
        self.head_dim = config.head_dim
        self.base = 10_000

        self.config = config

        # Compute inverse frequencies and register as buffer
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos_cache and sin_cache from position_ids.

        Args:
            position_ids: Position indices tensor

        Returns:
            cos_cache: Cosine cache tensor of shape (max_position_embeddings, head_dim // 2)
            sin_cache: Sine cache tensor of shape (max_position_embeddings, head_dim // 2)
        """
        device = position_ids.device

        # Ensure inv_freq is on the correct device
        inv_freq = self.inv_freq.to(device)

        # Create position embeddings for all possible positions up to max
        positions = torch.arange(self.max_seq_len_cached, dtype=torch.float32, device=device)

        # Compute the angles: outer product of positions and inv_freq
        freqs = torch.outer(positions, inv_freq)  # (max_position_embeddings, head_dim // 2)

        # Compute cos and sin caches
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        return cos_cache, sin_cache


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

        hidden_states = inputs_embeds

        # Compute cos_cache and sin_cache from position_ids
        cos_cache, sin_cache = self.rotary_emb(position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key=past_keys[i],
                past_value=past_values[i],
                cos_cache=cos_cache,
                sin_cache=sin_cache,
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

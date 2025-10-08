from __future__ import annotations

import torch
from torch import nn

from torch_onnx_models import _configs
from torch_onnx_models.components._attention_utils import create_attention_bias
from torch_onnx_models.components._decoder import DecoderLayer
from torch_onnx_models.components._rms_norm import RMSNorm
from torch_onnx_models.components._rotary_embedding import initialize_rope


class TextModel(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # embed tokens and positions
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(position_ids)

        # get the attention bias
        attention_bias = create_attention_bias(
            attention_mask=attention_mask,
            query_length=input_ids.shape[-1],
            dtype=hidden_states.dtype,
        )

        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values or [None] * len(self.layers)):
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values


class CausalLMModel(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
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

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess the state_dict to match the model's expected keys."""
        # For compatibility with HuggingFace models, we might need to rename some keys.
        if self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
            elif "model.embed_tokens.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        return state_dict

from __future__ import annotations

import copy

import torch
from torch import nn

from torch_onnx_models.components import apply_rms_norm, create_attention_bias, initialize_rope, Attention, MLP, RMSNorm
from torch_onnx_models._configs import ArchitectureConfig
from torch_onnx_models.models.base import CausalLMModel


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class Gemma3RMSNorm(RMSNorm):
    def forward(self, hidden_states):
        return apply_rms_norm(x=hidden_states.float(), weight=self.weight.float() + 1.0, eps=self.variance_epsilon).to(
            hidden_states.dtype
        )


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, rms_norm_class=Gemma3RMSNorm)
        self.mlp = MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class Gemma3AttentionBias(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sliding_window = config.sliding_window

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        or_mask_func: callable | None = None,
    ) -> dict[str, torch.Tensor]:
        attention_bias_dict = {
            "full_attention": create_attention_bias(input_ids=input_ids, attention_mask=attention_mask, dtype=dtype),
            "sliding_attention": create_attention_bias(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=dtype,
                sliding_window=self.sliding_window,
            ),
        }
        if or_mask_func is not None:
            or_mask = or_mask_func(input_ids, attention_mask)
            for key, value in attention_bias_dict.items():
                attention_bias_dict[key] = torch.where(or_mask, 0.0, value)
        return attention_bias_dict


class Gemma3TextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()

        # original model is in bf16 and the modeling code has this in bf16
        # huggingface chose to keep the same modeling code and use the weight dtype
        # but this would lead to different results if the weight is float32
        # we chose to use a constant in bf16 to match the original model
        # https://github.com/huggingface/transformers/pull/29402
        # https://github.com/huggingface/transformers/issues/38702
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16).item(),
        )
        self.attention_bias = Gemma3AttentionBias(config)

        self.layers = nn.ModuleList([Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        inputs_embeds: torch.Tensor | None = None,
        or_mask_func: callable | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # embed tokens and positions
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        position_embeddings_dict = {
            "full_attention": self.rotary_emb(position_ids),
            "sliding_attention": self.rotary_emb_local(position_ids),
        }

        # get the attention bias
        attention_bias_dict = self.attention_bias(
            input_ids, attention_mask, hidden_states.dtype, or_mask_func=or_mask_func
        )

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


class Gemma3CausalLMModel(CausalLMModel):
    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # override the model with Gemma3TextModel
        self.model = Gemma3TextModel(config)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess the state_dict to match the model's expected keys."""
        # we can use the language model weights from the multimodal model
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "multi_modal_projector" in key:
                state_dict.pop(key)
        return super().preprocess_weights(state_dict)

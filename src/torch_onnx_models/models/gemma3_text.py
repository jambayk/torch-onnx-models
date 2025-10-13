from __future__ import annotations

import copy

import torch
from torch import nn

from torch_onnx_models.components import (
    apply_rms_norm,
    apply_rotary_pos_emb,
    attention,
    create_attention_bias,
    initialize_rope,
    Attention,
    MLP,
    RMSNorm,
)
from torch_onnx_models._configs import ArchitectureConfig
from torch_onnx_models.models.base import CausalLMModel


class Gemma3RMSNorm(RMSNorm):
    def forward(self, hidden_states):
        return apply_rms_norm(x=hidden_states.float(), weight=self.weight.float() + 1, eps=self.variance_epsilon).to(
            hidden_states.dtype
        )


class Gemma3Attention(Attention):
    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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

        if self.q_norm is not None and self.k_norm is not None:
            input_shape = hidden_states.shape[:-1]
            query_states = self.q_norm(query_states.view(*input_shape, -1, self.head_dim))
            key_states = self.k_norm(key_states.view(*input_shape, -1, self.head_dim))
            query_states = query_states.view(*input_shape, -1)
            key_states = key_states.view(*input_shape, -1)

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


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config)
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
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # embed tokens and positions
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        position_embeddings_dict = {
            "full_attention": self.rotary_emb(position_ids),
            "sliding_attention": self.rotary_emb_local(position_ids),
        }

        # get the attention bias
        attention_bias_dict = {
            "full_attention": create_attention_bias(
                input_ids=input_ids, attention_mask=attention_mask, dtype=hidden_states.dtype
            ),
            "sliding_attention": create_attention_bias(
                input_ids=input_ids,
                attention_mask=attention_mask,
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


class Gemma3CausalLMModel(CausalLMModel):
    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # override the model with Gemma3TextModel
        self.model = Gemma3TextModel(config)


def create_image_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_id: int = 262144
) -> torch.Tensor:
    """
    Create a mask where the image tokens attend to all tokens within the same image.

    Args:
        input_ids (torch.Tensor): The input token IDs of shape (batch_size, query_length).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, total_length).
        image_token_id (int, optional): The token ID that represents an image token. Defaults to 262144.

    Returns:
        torch.Tensor: A mask of shape (batch_size, 1, query_length, total_length) where True indicates that the query token can attend to the key token.
    """
    query_length = input_ids.shape[-1]
    batch_size, kv_length = attention_mask.shape

    is_img = input_ids == image_token_id
    leading_zero = torch.zeros((batch_size, 1), dtype=is_img.dtype)
    prev = torch.cat([leading_zero, is_img[:, :-1]], dim=1)
    starts = is_img & ~prev
    gid = torch.cumsum(starts, dim=1)
    gid = torch.where(is_img, gid, torch.full_like(gid, 0))

    q_gid = gid
    # just like attention mask bias, the exporter needs to provide some past kvs, otherwise this becomes a no-op in the exported model
    # even if we don't use cat and use some other data dependent slicing, torch export might think query_length == kv_length
    k_gid = torch.cat(
        [
            torch.full((batch_size, kv_length - query_length), -1, dtype=gid.dtype),
            gid,
        ],
        dim=1,
    )
    mask = (q_gid.unsqueeze(2) == k_gid.unsqueeze(1)) & (q_gid.unsqueeze(2) > 0)
    return mask.unsqueeze(1)

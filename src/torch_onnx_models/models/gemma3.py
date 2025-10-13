from __future__ import annotations

import re
from functools import partial

import torch
from torch import nn

from torch_onnx_models._configs import ArchitectureConfig
from torch_onnx_models.models.gemma3_text import Gemma3TextModel, Gemma3RMSNorm
from torch_onnx_models.models.sigslip import SiglipVisionModel


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs


def create_image_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """
    Create a mask where the image tokens attend to all tokens within the same image.

    Args:
        input_ids (torch.Tensor): The input token IDs of shape (batch_size, query_length).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, total_length).
        image_token_id (int): The token ID that represents an image token.

    Returns:
        torch.Tensor: A mask of shape (batch_size, 1, query_length, total_length) where True indicates that the query token can attend to the key token.
    """
    query_length = input_ids.shape[-1]
    batch_size, kv_length = attention_mask.shape

    is_img = input_ids == image_token_id
    leading_zero = torch.zeros((batch_size, 1), dtype=is_img.dtype, device=is_img.device)
    # TODO(jambayk): maybe try a different way to slice since it's creates unnecessary ops in the graph
    prev = torch.cat([leading_zero, is_img[:, :-1]], dim=1)
    starts = is_img & ~prev
    gid = torch.cumsum(starts, dim=1)
    # full_like creates CastLike which might not be supported in some runtimes?
    gid = torch.where(is_img, gid, torch.full_like(gid, 0))

    q_gid = gid
    # just like attention mask bias, the exporter needs to provide some past kvs, otherwise this becomes a no-op in the exported model
    # even if we don't use cat and use some other data dependent slicing, torch export might think query_length == kv_length
    k_gid = torch.cat(
        [
            torch.full((batch_size, kv_length - query_length), -1, dtype=gid.dtype, device=gid.device),
            gid,
        ],
        dim=1,
    )
    mask = (q_gid.unsqueeze(2) == k_gid.unsqueeze(1)) & (q_gid.unsqueeze(2) > 0)
    return mask.unsqueeze(1)


# need a better name for this module
class Gemma3MultiModalMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_token_id = config.image_token_id

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, image_features: torch.Tensor):
        special_image_mask = (input_ids == self.image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(special_image_mask, image_features)


class Gemma3MultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.mixer = Gemma3MultiModalMixer(config)
        self.language_model = Gemma3TextModel(ArchitectureConfig.from_transformers(config.text_config))
        self.image_token_id = config.image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        pixel_values: torch.Tensor | None = None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            vision_outputs = self.vision_tower(pixel_values=pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)
            inputs_embeds = self.mixer(input_ids, inputs_embeds, image_features)

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            or_mask_func=partial(create_image_mask, image_token_id=self.image_token_id),
        )


class Gemma3ConditionalGenerationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3MultiModalModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        pixel_values: torch.Tensor | None = None,
    ):
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
        )
        logits = self.lm_head(hidden_states)
        return logits, present_key_values

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess the state_dict to match the model's expected keys."""
        key_map = {
            r"^language_model\.model": "model.language_model",
            r"^vision_tower": "model.vision_tower",
            r"^multi_modal_projector": "model.multi_modal_projector",
            r"^language_model\.lm_head": "lm_head",
        }
        for key in list(state_dict.keys()):
            for pattern, replacement in key_map.items():
                new_key = re.sub(pattern, replacement, key)
                if new_key != key:
                    state_dict[new_key] = state_dict.pop(key)
                    break

        if self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["model.language_model.embed_tokens.weight"] = state_dict["lm_head.weight"]
            elif "model.language_model.embed_tokens.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"]
        return state_dict

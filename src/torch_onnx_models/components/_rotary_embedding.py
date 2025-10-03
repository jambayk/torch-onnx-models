from __future__ import annotations

import math

import torch
from torch import nn

from torch_onnx_models import _configs
from torch_onnx_models.components._rotary_embedding_utils import get_rotary_pos_emb


def _get_default_inv_freq(config: _configs.ArchitectureConfig) -> torch.Tensor:
    return 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim)
    )


def _get_cos_sin_cache(
    max_position_embeddings: int, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # should we do max position embeddings or original max position embeddings?
    # some models like llama4 has 10 million
    pos = torch.arange(0, max_position_embeddings, dtype=torch.float)
    angles = torch.outer(pos, inv_freq)
    # probably need to cast the caches to the same dtype as the model
    return torch.cos(angles), torch.sin(angles)


class BaseRope(nn.Module):
    def _register_cos_sin_cache(self, cos: torch.Tensor, sin: torch.Tensor):
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return get_rotary_pos_emb(position_ids, self.cos_cache, self.sin_cache)


class DefaultRope(BaseRope):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            inv_freq = _get_default_inv_freq(config)
            cos_cache, sin_cache = _get_cos_sin_cache(
                config.max_position_embeddings, inv_freq
            )
            self._register_cos_sin_cache(cos_cache, sin_cache)


class Llama3Rope(BaseRope):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            inv_freq = _get_default_inv_freq(config)

            # https://github.com/huggingface/transformers/blob/9f2d5666f8fda5b647e5f64dfc8ba1edd7a87a1e/src/transformers/modeling_rope_utils.py#L497
            factor = config.rope_scaling["factor"]
            low_freq_factor = config.rope_scaling["low_freq_factor"]
            high_freq_factor = config.rope_scaling["high_freq_factor"]
            old_context_len = config.rope_scaling["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
            )
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            inv_freq_llama = torch.where(
                is_medium_freq, smoothed_inv_freq, inv_freq_llama
            )

            cos_cache, sin_cache = _get_cos_sin_cache(
                config.max_position_embeddings, inv_freq_llama
            )
            self._register_cos_sin_cache(cos_cache, sin_cache)


def initialize_rope(config: _configs.ArchitectureConfig) -> nn.Module:
    if config.rope_type == "default":
        return DefaultRope(config)
    if config.rope_type == "llama3":
        return Llama3Rope(config)

    raise ValueError(f"Unsupported rope type: {config.rope_type}")

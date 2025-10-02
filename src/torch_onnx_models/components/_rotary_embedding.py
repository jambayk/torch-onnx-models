from __future__ import annotations

import math

import torch
from torch import nn

from torch_onnx_models import _configs
from torch_onnx_models.components._rotary_embedding_utils import get_rotary_pos_emb


class Llama3Rope(nn.Module):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim))

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
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

            # should we do max position embeddings or original max position embeddings?
            # some models like llama4 has 10 million
            pos = torch.arange(old_context_len, dtype=torch.float)
            angles = torch.outer(pos, inv_freq_llama)

            # probably need to cast the caches to the same dtype as the model
            # get it from the config?
            self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
            self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return get_rotary_pos_emb(position_ids, self.cos_cache, self.sin_cache)

def initialize_rope(config: _configs.ArchitectureConfig) -> nn.Module:
    if config.rope_type == "llama3":
        return Llama3Rope(config)

    raise ValueError(f"Unsupported rope type: {config.rope_type}")

from __future__ import annotations

import torch
from torch_onnx_models import _configs


def create_rope_caches(
    config: _configs.ArchitectureConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Initialize rope frequencies using the utility function

    inv_freq, attention_factor = _initialize_rope_freqs(config=config)

    # Create position indices for all positions up to max_position_embeddings
    max_seq_len = config.max_position_embeddings
    # TODO: position_scale
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute the angles for each position and frequency
    # positions: [max_seq_len], inv_freq: [dim//2] -> angles: [max_seq_len, dim//2]
    angles = torch.outer(positions, inv_freq)

    # Precompute cos and sin caches
    cos_cache = torch.cos(angles) * attention_factor
    sin_cache = torch.sin(angles) * attention_factor
    return cos_cache, sin_cache


def _initialize_rope_freqs(
    config: _configs.ArchitectureConfig,
) -> tuple[torch.Tensor, float]:
    if config.rope_type == "default":
        return _compute_default_rope_parameters(config)
    raise ValueError(f"Unsupported rope_type: {config.rope_type}")


def _compute_default_rope_parameters(
    config: _configs.ArchitectureConfig,
) -> tuple[torch.Tensor, float]:
    # https://github.com/huggingface/transformers/blob/6dc9ed87a02db8b4ecc26a5e98596cd2bba380b5/src/transformers/modeling_rope_utils.py#L92
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor
    head_dim = config.head_dim
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq, attention_factor


# TODO(jambayk): add support for interleaved format if needed
# requires torch 2.9+
def apply_rope(
    x: torch.Tensor,
    *,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    This function modifies the input hidden states by applying the RoPE transformation
    using the provided cosine and sine caches based on the given position IDs.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        torch.Tensor: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    return torch.onnx.ops.rotary_embedding(
        x,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        rotary_embedding_dim=rotary_embedding_dim,
        num_heads=num_heads,
    )


def apply_rope_decomposed(
    *,
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    This function modifies the input hidden states by applying the RoPE transformation
    using the provided cosine and sine caches based on the given position IDs.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        torch.Tensor: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    batch_size, seq_length, _ = x.shape
    x = x.reshape(batch_size, seq_length, num_heads, -1)
    # doing conditionals so that the graph is cleaner when rotary_embedding_dim is 0
    if rotary_embedding_dim == 0:
        x_rot, x_pass = x, None
    else:
        x_rot, x_pass = x[..., :rotary_embedding_dim], x[..., rotary_embedding_dim:]

    cos = cos_cache[position_ids].unsqueeze(2)
    sin = sin_cache[position_ids].unsqueeze(2)

    x1, x2 = x_rot.chunk(2, dim=-1)

    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    x_applied = torch.cat((real, imag), dim=-1)

    if x_pass is not None:
        return torch.cat([x_applied, x_pass], dim=-1)

    return x_applied.reshape(batch_size, seq_length, -1)


def apply_rope_contrib(
    *,
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    This function modifies the input hidden states by applying the RoPE transformation
    using the provided cosine and sine caches based on the given position IDs.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, head_dim).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, head_dim).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        torch.Tensor: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    return torch.onnx.ops.symbolic(
        "com.microsoft::RotaryEmbedding",
        [x, position_ids, cos_cache, sin_cache],
        attrs={"num_heads": num_heads, "rotary_embedding_dim": rotary_embedding_dim},
        dtype=x.dtype,
        shape=x.shape,
        version=1,
    )

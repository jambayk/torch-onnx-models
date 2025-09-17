from __future__ import annotations

import torch


def initialize_rope_freqs(
    dim: int, max_position_embeddings: int, base: int = 10_000
) -> tuple[torch.Tensor, torch.Tensor]:
    # just a place holder that returns random values for now
    return torch.rand(max_position_embeddings, dim), torch.rand(max_position_embeddings, dim)


def apply_rope(
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
        x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_length, head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, head_dim).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, head_dim).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0, meaning full embedding).

    Returns:
        torch.Tensor: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    batch_size, _, seq_length, head_dim = x.shape
    # there is a bug in the implementation of the torch.onnx.ops.rotary_embedding
    # reshape to 3D shape (batch_size, seq_length, num_heads * head_dim)
    x = x.transpose(1, 2).contiguous().reshape(batch_size, seq_length, num_heads * head_dim)
    x = torch.onnx.ops.rotary_embedding(
        x,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        rotary_embedding_dim=rotary_embedding_dim,
        num_heads=num_heads,
    )
    return x.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2).contiguous()


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
        x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_length, head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, rotary_embedding_dim // 2).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        torch.Tensor: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    # doing conditionals so that the graph is cleaner when rotary_embedding_dim is 0
    if rotary_embedding_dim == 0:
        x_rot, x_pass = x, None
    else:
        x_rot, x_pass = x[..., :rotary_embedding_dim], x[..., rotary_embedding_dim:]

    cos = cos_cache[position_ids].unsqueeze(1)
    sin = sin_cache[position_ids].unsqueeze(1)

    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]

    real = x_even * cos - x_odd * sin
    imag = x_even * sin + x_odd * cos

    x_applied = torch.cat((real, imag), dim=-1)

    if x_pass is not None:
        return torch.cat([x_applied, x_pass], dim=-1)

    return x_applied


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
        x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_length, head_dim).
        cos_cache (torch.Tensor): The cosine cache tensor of shape (max_position_embeddings, head_dim).
        sin_cache (torch.Tensor): The sine cache tensor of shape (max_position_embeddings, head_dim).
        position_ids (torch.Tensor): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0, meaning full embedding).

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

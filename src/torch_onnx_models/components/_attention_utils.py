from __future__ import annotations

import torch
import torch.nn as nn


# TODO(jambayk): generalize to include sliding window
def create_attention_bias(
    *, attention_mask: torch.Tensor, query_length: torch.Tensor, dtype: torch.dtype, mask_value: float = None
) -> torch.Tensor:
    """
    Create attention bias for use in attention mechanisms.

    Args:
        attention_mask (torch.Tensor): The attention mask tensor.
        query_length (torch.Tensor): The length of the query sequence.
        dtype (torch.dtype): The desired data type for the output tensor.

    Returns:
        torch.Tensor: The attention bias tensor reshaped and cast to the specified dtype.
    """
    all_indices = attention_mask.cumsum(-1)
    kv_indices = all_indices[:, None, :]
    # should we make this not data dependent slicing?
    # like q_indices = torch.arange(Q, device=attention_mask.device)
    q_indices = all_indices[:, -query_length:, None]
    full_mask = q_indices >= kv_indices
    full_mask &= attention_mask[:, None, :].to(torch.bool)
    # make the negative value configurable
    mask_value = torch.finfo(dtype).min if mask_value is None else torch.tensor(mask_value, dtype=dtype)
    return torch.where(full_mask, 0.0, mask_value)[:, None, :, :]


# TODO(jambayk): add doc strings for shape of outputs
# should we work on 3d inputs instead. ops like Attention, GroupQueryAttention, MultiHeadAttention, RotaryEmbedding seem to be
# optimized for 3d inputs of shape (batch_size, seq_length, hidden_size)
# otherwise, there is a lot of transposes and we probably would need to do graph surgery to eliminate them.
# Hardest would be to change the shape of the kv cache inputs/outputs
# but the decomposed attention function would have multiple reshapes inside
# should we make scale optional? maybe not since it requires us to get the head_dim from the input shape
def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (torch.Tensor): The attention bias tensor of shape (batch_size, 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
    """
    return torch.onnx.ops.attention(
        query, key, value, bias, past_key, past_value, kv_num_heads=kv_num_heads, q_num_heads=q_num_heads, scale=scale
    )[:3]


def _reshape_3d_to_4d(x: torch.Tensor, batch_size: int, seq_length: int, num_heads: int) -> torch.Tensor:
    return x.reshape(batch_size, seq_length, num_heads, -1).transpose(1, 2).contiguous()


def _prepare_qkv_mha(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    batch_size: int,
    seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare key and value tensors for Multi-Head Attention (MHA) operation.

    Args:
        key (torch.Tensor): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the prepared key, value, present key, and present value tensors.
    """
    key = _reshape_3d_to_4d(key, batch_size, seq_length, kv_num_heads)
    value = _reshape_3d_to_4d(value, batch_size, seq_length, kv_num_heads)

    # TODO(jambayk): put some guidance that there should not be data-dependent conditionals in general but None checks are ok
    if past_key is not None and past_value is not None:
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    present_key = key
    present_value = value

    if q_num_heads != kv_num_heads:
        key = key.repeat_interleave(q_num_heads // kv_num_heads, dim=1)
        value = value.repeat_interleave(q_num_heads // kv_num_heads, dim=1)
    return key, value, present_key, present_value


def attention_decomposed(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (torch.Tensor): The attention bias tensor of shape (batch_size, 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
    """
    batch_size, seq_length, _ = query.shape
    query = _reshape_3d_to_4d(query, batch_size, seq_length, q_num_heads)
    key, value, present_key, present_value = _prepare_kv_mha(
        key=key,
        value=value,
        past_key=past_key,
        past_value=past_value,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    attn_weight = torch.matmul(query, key.transpose(2, 3)) * scale
    attn_weight = attn_weight + bias

    attn_weights = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
    return attn_output, present_key, present_value


def attention_contrib_mha(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (torch.Tensor): The attention bias tensor of shape (batch_size, 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
    """
    batch_size, seq_length, _ = query.shape
    key, value, present_key, present_value = _prepare_kv_mha(
        key=key,
        value=value,
        past_key=past_key,
        past_value=past_value,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    return (
        torch.onnx.ops.symbolic(
            "com.microsoft::MultiHeadAttention",
            [query, key, value, None, None, bias],
            attrs={"num_heads": q_num_heads, "scale": scale},
            dtype=value.dtype,
            shape=(batch_size, seq_length, q_num_heads * value.shape[-1]),
            version=1,
        )
        present_key,
        present_value,
    )

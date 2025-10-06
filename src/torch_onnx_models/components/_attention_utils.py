from __future__ import annotations

import torch
from torch import nn


# TODO(jambayk): generalize to include sliding window
def create_attention_bias(
    *,
    attention_mask: torch.Tensor,
    query_length: int | torch.SymInt,
    dtype: torch.dtype,
    sliding_window: int | None = None,
    mask_value: float | None = None,
) -> torch.Tensor:
    """
    Create attention bias for use in attention mechanisms.

    Args:
        attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, total_length).
        query_length (torch.Tensor): The length of the query sequence.
        dtype (torch.dtype): The desired data type for the output tensor.
        sliding_window (int, optional): The size of the sliding window for local attention. If None, full attention is used.
        mask_value (float, optional): The value to use for masked positions. If None, uses the minimum value for the specified dtype.

    Returns:
        torch.Tensor: The attention bias tensor reshaped and cast to the specified dtype of shape (batch_size, 1, query_length, total_length).
    """
    assert attention_mask.dim() == 2, (
        "attention_mask should be of shape (batch_size, total_length)"
    )
    all_indices = attention_mask.cumsum(-1)
    kv_indices = torch.unsqueeze(all_indices, 1)
    # should we make this not data dependent slicing?
    # like q_indices = torch.arange(query_length, device=attention_mask.device)
    q_indices = all_indices[:, -query_length:]
    q_indices = torch.unsqueeze(q_indices, -1)
    full_mask = q_indices >= kv_indices
    if sliding_window is not None:
        full_mask = torch.logical_and(full_mask, q_indices - kv_indices < sliding_window)
    full_mask = torch.logical_and(
        torch.unsqueeze(attention_mask, 1).to(torch.bool), full_mask
    )
    # make the negative value configurable
    mask_value = torch.finfo(dtype).min if mask_value is None else mask_value
    return torch.unsqueeze(torch.where(full_mask, 0.0, mask_value), 1)


# requires latest nightly ort to run inference correctly on exported model
# GQA case is incorrect in stable releases
def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    # rename back to attention_mask?
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
        bias (torch.Tensor): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
            attention_output (torch.Tensor): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (torch.Tensor): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (torch.Tensor): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
    """
    if torch.onnx.is_in_onnx_export():
        present_key_shape = (
            past_key.shape[0],
            past_key.shape[1],
            past_key.shape[2] + query.shape[1],
            past_key.shape[3],
        )
        present_value_shape = (
            past_value.shape[0],
            past_value.shape[1],
            past_value.shape[2] + query.shape[1],
            past_value.shape[3],
        )
        return torch.onnx.ops.symbolic_multi_out(
            "Attention",
            [query, key, value, bias, past_key, past_value],
            attrs=dict(kv_num_heads=kv_num_heads, q_num_heads=q_num_heads, scale=scale),
            dtypes=(query.dtype, key.dtype, value.dtype),
            shapes=(query.shape, present_key_shape, present_value_shape),
            version=23,
        )
    # TODO(justinchuby): Unfortunately, the meta implementation of torch.onnx.ops.attention
    # is using torch sdpa which has a strict requirement on the input shapes. Will fix that later
    # Maybe I set the input shapes wrong
    return torch.onnx.ops.attention(
        query,
        key,
        value,
        bias,
        past_key,
        past_value,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        scale=scale,
    )[:3]


def _reshape_3d_to_4d(
    x: torch.Tensor, batch_size: int, seq_length: int, num_heads: int
) -> torch.Tensor:
    """
    Reshape a 3D tensor to a 4D tensor for multi-head attention.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        batch_size (int): The batch size.
        seq_length (int): The sequence length.
        num_heads (int): The number of attention heads.

    Returns:
        torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_length, head_dim).
    """
    return x.reshape(batch_size, seq_length, num_heads, -1).transpose(1, 2).contiguous()


def _prepare_kv_mha(
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
            key (torch.Tensor): The prepared key tensor of shape (batch_size, q_num_heads, total_length, head_dim).
            value (torch.Tensor): The prepared value tensor of shape (batch_size, q_num_heads, total_length, head_dim).
            present_key (torch.Tensor): The present key tensor for caching of shape (batch_size, kv_num_heads, total_length, head_dim).
            present_value (torch.Tensor): The present value tensor for caching of shape (batch_size, kv_num_heads, total_length, head_dim).
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
        bias (torch.Tensor): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
            attention_output (torch.Tensor): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (torch.Tensor): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (torch.Tensor): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
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
    if torch.onnx.is_in_onnx_export():
        # export is failing due to shape mismatch which shouldn't be happening
        attn_weight = torch.onnx.ops.symbolic(
            "Add",
            [attn_weight, bias],
            attrs={},
            dtype=attn_weight.dtype,
            shape=attn_weight.shape,
        )
    else:
        attn_weight = attn_weight + bias

    attn_weights = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = (
        attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
    )
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
        bias (torch.Tensor): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (torch.Tensor | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (torch.Tensor | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the attention output, present key, and present value.
            attention_output (torch.Tensor): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (torch.Tensor): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (torch.Tensor): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
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
        ),
        present_key,
        present_value,
    )

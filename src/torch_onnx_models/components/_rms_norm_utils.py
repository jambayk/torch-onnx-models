from __future__ import annotations

import torch
from torch import nn


# this uses the float32 as the data type until the final multiplication with weight
# TODO(jambayk): expose dtype as an argument if needed
def apply_rms_norm(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (torch.Tensor): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        torch.Tensor: The normalized hidden states with the same shape as input.
    """
    # This will produce the correct ONNX standard ops based on the opset requested
    # assumes opset 23 will be used during export
    # rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
    return torch.ops.aten.rms_norm(x, (x.size(-1),), weight, eps)


def apply_rms_norm_decomposed(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (torch.Tensor): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        torch.Tensor: The normalized hidden states with the same shape as input.
    """
    x_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(x_dtype)


def apply_rms_norm_contrib(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (torch.Tensor): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        torch.Tensor: The normalized hidden states with the same shape as input.
    """
    return torch.onnx.ops.symbolic(
        # SimplifiedLayerNormalization is a contrib op but it is miscongured as ai.onnx in ORT
        "ai.onnx::SimplifiedLayerNormalization",
        [x, weight],
        attrs={"epsilon": eps},
        dtype=x.dtype,
        shape=x.shape,
        version=1,
    )

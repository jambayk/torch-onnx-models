from __future__ import annotations


import torch
from torch_onnx_models import _configs


def _create_example_inputs(
    config: _configs.ArchitectureConfig, export_config: _configs.ExportConfig
):
    """Create example inputs and dynamic axes for ONNX export."""
    num_hidden_layers = config.num_hidden_layers
    batch = "batch"
    sequence_len = "sequence_len"
    past_sequence_len = "past_sequence_len"

    dynamic_shapes = {
        "input_ids": {0: batch, 1: sequence_len},
        "attention_mask": {
            0: batch,
            1: "past_sequence_len+sequence_len",
        },
        "position_ids": {
            0: batch,
            1: sequence_len,
        },
        "past_key_values": [
            [{0: batch, 2: past_sequence_len} for _ in range(num_hidden_layers)],
            [{0: batch, 2: past_sequence_len} for _ in range(num_hidden_layers)],
        ],
    }

    example_batch_size = 2
    example_past_sequence_len = 2
    example_sequence_len = 3
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    example_inputs = dict(
        input_ids=torch.randint(
            0, 2, (example_batch_size, example_sequence_len), dtype=torch.int64
        ),
        attention_mask=torch.ones(
            (example_batch_size, example_past_sequence_len + example_sequence_len),
            dtype=torch.int64,
        ),
        position_ids=torch.arange(
            example_past_sequence_len,
            example_past_sequence_len + example_sequence_len,
            dtype=torch.int64,
        ).expand((example_batch_size, -1)),
        past_key_values=[
            (
                torch.randn(
                    example_batch_size,
                    num_key_value_heads,
                    example_sequence_len,
                    head_dim,
                ),
                torch.randn(
                    example_batch_size,
                    num_key_value_heads,
                    example_sequence_len,
                    head_dim,
                ),
            )
            for _ in range(num_hidden_layers)
        ],
    )

    return example_inputs, dynamic_shapes

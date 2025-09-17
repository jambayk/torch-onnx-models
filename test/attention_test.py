import unittest

import torch
from torch_onnx_models.components._attention import Attention
from torch_onnx_models.components._attention_utils import create_attention_bias
from argparse import Namespace


class AttentionTest(unittest.TestCase):
    def test_export(self):

        config = Namespace(
            hidden_size=2048, head_dim=64, num_attention_heads=32, num_key_value_heads=8, attention_bias=False
        )

        attention = Attention(config)

        batch_size = 1
        past_length = 5
        seq_length = 10
        max_length = 100
        attention_mask = torch.ones((batch_size, past_length + seq_length), dtype=torch.bool)

        inputs = {
            "hidden_states": torch.randn(batch_size, seq_length, config.hidden_size),
            "attention_bias": create_attention_bias(
                attention_mask=attention_mask, query_length=seq_length, dtype=torch.float32
            ),
            "position_ids": torch.arange(past_length, past_length + seq_length).unsqueeze(0),
            "cos_cache": torch.randn(max_length, config.head_dim // 2),
            "sin_cache": torch.randn(max_length, config.head_dim // 2),
            "past_key": torch.randn(batch_size, config.num_key_value_heads, past_length, config.head_dim),
            "past_value": torch.randn(batch_size, config.num_key_value_heads, past_length, config.head_dim),
        }
        dynamic_shapes = {
            # "hidden_states": {0: "batch_size", 1: "seq_length"},
            # "attention_bias": {0: "batch_size", 2: "seq_length", 3: "total_seq_length"},
            # "position_ids": {0: "batch_size", 1: "seq_length"},
            # "cos_cache": {0: "max_seq_length"},
            # "sin_cache": {0: "max_seq_length"},
            # "past_key": {0: "batch_size", 2: "past_seq_length"},
            # "past_value": {0: "batch_size", 2: "past_seq_length"},
        }

        onnx_program = torch.onnx.export(
            attention,
            tuple(inputs.values()),
            input_names=list(inputs.keys()),
            output_names=["attn_output", "present_key", "present_value"],
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
            dynamo=True,
        )


if __name__ == "__main__":
    unittest.main()

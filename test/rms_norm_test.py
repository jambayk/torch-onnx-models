import unittest

import parameterized
import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch_onnx_models.components._rms_norm import RMSNorm


class RMSNormTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("ort", 18),
            ("onnx", 18),
            ("onnx", 23),
        ]
    )
    def test_export(self, mode: str, opset_version: int):
        hidden_size = 768
        model = RMSNorm(hidden_size=hidden_size, mode=mode)
        model.eval()

        input = torch.randn(2, 128, 768)

        onnx_program = torch.onnx.export(
            model,
            (input,),
            opset_version=opset_version,
            dynamic_shapes={"hidden_states": {0: "batch_size", 1: "seq_len"}},
            dynamo=True,
            verbose=False,
        )
        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    unittest.main()

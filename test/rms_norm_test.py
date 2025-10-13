import unittest

import parameterized
import torch
from torch.onnx._internal.exporter import _testing as onnx_testing

from torch_onnx_models.components._rms_norm import RMSNorm


class RMSNormTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (18,),
            (23,),
        ]
    )
    def test_export(self, opset_version: int):
        hidden_size = 768
        model = RMSNorm(hidden_size=hidden_size)
        model.eval()

        input = torch.randn(2, 128, 768)

        ref_model = RMSNorm(hidden_size=hidden_size)
        ref_model.eval()

        ref_ep = torch.export.export(ref_model, (input,))

        onnx_program = torch.onnx.export(
            model,
            (input,),
            opset_version=opset_version,
            dynamic_shapes={"hidden_states": {0: "batch_size", 1: "seq_len"}},
            dynamo=True,
            verbose=False,
        )

        onnx_program.exported_program = ref_ep
        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    unittest.main()

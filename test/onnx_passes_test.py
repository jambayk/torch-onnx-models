from __future__ import annotations

import unittest

import torch
from torch_onnx_models import onnx_passes
from torch_onnx_models.components._rms_norm import RMSNorm


class AssignNamesPassTest(unittest.TestCase):
    def test_pass(self):
        class Model(torch.nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.rms_norm = RMSNorm(hidden_size=hidden_size, mode="onnx")

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.rms_norm(hidden_states)

        hidden_size = 768
        model = Model(hidden_size=hidden_size)
        model.eval()

        input = torch.randn(2, 128, 768)

        onnx_program = torch.onnx.export(
            model,
            (input,),
            opset_version=18,
            dynamic_shapes={"hidden_states": {0: "batch_size", 1: "seq_len"}},
            dynamo=True,
            verbose=False,
        )
        print(onnx_program.model)
        pass_result = onnx_passes.AssignNamesPass()(onnx_program.model)
        print(pass_result.model)
        # TODO: Actually assert some things


if __name__ == "__main__":
    unittest.main()

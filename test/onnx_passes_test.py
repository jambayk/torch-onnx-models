from __future__ import annotations

import unittest

import torch
from torch_onnx_models import onnx_passes, _barrier
from torch_onnx_models.components._rms_norm import RMSNorm


class AssignNamesPassTest(unittest.TestCase):
    def test_pass(self):
        class Model(torch.nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.rms_norm = RMSNorm(hidden_size=hidden_size)

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


class RemoveBarrierPassTest(unittest.TestCase):
    def setUp(self) -> None:
        _barrier.ENABLE_BARRIER = True

    def tearDown(self) -> None:
        _barrier.ENABLE_BARRIER = False

    def test_pass(self):
        x = torch.randn(2, 3)
        onnx_program = torch.onnx.export(
            QuickGELUActivation(), (x,), dynamo=True, verbose=False
        )
        model = onnx_program.model
        self.assertIn("Barrier", [node.op_type for node in model.graph])
        onnx_passes.RemoveBarrierPass()(model)
        self.assertEqual(len(model.graph), 3)
        self.assertNotIn("Barrier", [node.op_type for node in model.graph])


class SubgraphReplacementTest(unittest.TestCase):
    def test_pass(self):
        x = torch.randn(2, 3)
        model = torch.onnx.export(
            QuickGELUActivation(), (x,), dynamo=True, verbose=False
        ).model
        ort_model = torch.onnx.export(
            MsftQuickGELUActivation(), (x,), dynamo=True, verbose=False
        ).model
        region_start = next(node for node in model.graph if node.op_type == "Barrier")
        region_end = next(
            node for node in reversed(model.graph) if node.op_type == "Barrier"
        )
        self.assertNotIn("QuickGelu", [node.op_type for node in model.graph])
        onnx_passes.replace_subgraph(
            model, region_start.inputs, region_end.outputs, ort_model.graph
        )
        self.assertIn("QuickGelu", [node.op_type for node in model.graph])
        print(model)

        onnx_passes.RemoveBarrierPass()(model)
        self.assertEqual(len(model.graph), 1)
        print(model)


class RemoveBarrierPassTest(unittest.TestCase):
    def test_pass(self):
        x = torch.randn(2, 3)
        onnx_program = torch.onnx.export(
            QuickGELUActivation(), (x,), dynamo=True, verbose=False
        )
        model = onnx_program.model
        self.assertIn("Barrier", [node.op_type for node in model.graph])
        onnx_passes.RemoveBarrierPass()(model)
        self.assertEqual(len(model.graph), 3)
        self.assertNotIn("Barrier", [node.op_type for node in model.graph])


if __name__ == "__main__":
    unittest.main()

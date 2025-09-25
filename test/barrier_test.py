import unittest

import torch

from torch_onnx_models import _barrier
from torch_onnx_models.components import _activations


@_barrier.with_barrier(
    metadata={
        "op": "domain::MyCustomOp",
        "additional_attributes": {"attr1": 123, "attr2": "value"},
    }
)
def decorated_func(x, y, *, z="default"):
    return x * 2 + y


class ModelWithDecoratedBarrier(torch.nn.Module):
    def forward(self, x, y):
        return decorated_func(x, y)


class Model(torch.nn.Module):
    def forward(self, x, y, z="default"):
        identifier = _barrier.get_identifier("my_region")
        x, y = _barrier.barrier_op(
            (x, y), {"z": z}, region_identifier=identifier, type="input"
        )
        result = x * 2 + y
        result = _barrier.barrier_op(
            (result,), region_identifier=identifier, type="output"
        )
        return result


class BarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        _barrier.ENABLE_BARRIER = True

    def tearDown(self) -> None:
        _barrier.ENABLE_BARRIER = False

    def test_decorator(self):
        model = ModelWithDecoratedBarrier().eval()

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        onnx_program = torch.onnx.export(model, (x, y), dynamo=True, verbose=False)
        nodes = [node.op_type for node in onnx_program.model.graph]
        self.assertEqual(nodes.count("Barrier"), 2)
        onnx_program.save("barrier_decorator.onnx")

    def test_annotations(self):
        model = Model().eval()

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        onnx_program = torch.onnx.export(model, (x, y), dynamo=True, verbose=False)
        nodes = [node.op_type for node in onnx_program.model.graph]
        self.assertEqual(nodes.count("Barrier"), 2)
        barrier_node = next(
            node for node in onnx_program.model.graph if node.op_type == "Barrier"
        )
        attributes = _barrier.get_attrs(barrier_node)
        self.assertEqual(attributes, {"z": "default"})

    def test_quick_gelu(self):
        model = _activations.QuickGELUActivation()
        x = torch.randn(2, 3)
        onnx_program = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
        barrier_node = next(
            node for node in onnx_program.model.graph if node.op_type == "Barrier"
        )
        attributes = _barrier.get_attrs(barrier_node)
        self.assertEqual(attributes, {})
        metadata = _barrier.get_metadata(barrier_node)
        self.assertEqual(metadata, {"region": "quick_gelu"})


if __name__ == "__main__":
    unittest.main()

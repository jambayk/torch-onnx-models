import unittest

import torch

from torch_onnx_models.components import _barrier


@_barrier.with_barrier(
    metadata={
        "op": "domain::MyCustomOp",
        "attributes": {"attr1": 123, "attr2": "value"},
    }
)
def decorated_func(x, y):
    return x * 2 + y


class ModelWithDecoratedBarrier(torch.nn.Module):
    def forward(self, x, y):
        return decorated_func(x, y)


class Model(torch.nn.Module):
    def forward(self, x, y, z="default"):
        identifier = _barrier.get_identifier("my_region")
        x, y = _barrier.barrier_op(
            (x, y), {"z": z}, group_identifier=identifier, type="input"
        )
        result = x * 2 + y
        result = _barrier.barrier_op(
            (result,), group_identifier=identifier, type="output"
        )
        return result


class BarrierTest(unittest.TestCase):
    def test_decorator(self):
        model = ModelWithDecoratedBarrier().eval()

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        onnx_program = torch.onnx.export(
            model,
            (x, y),
            dynamo=True,
        )
        nodes = [node.op_type for node in onnx_program.model.graph]
        self.assertEqual(nodes.count("Barrier"), 2)
        onnx_program.save("barrier_decorator.onnx")

    def test_annotations(self):
        model = Model().eval()

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        onnx_program = torch.onnx.export(
            model,
            (x, y),
            dynamo=True,
        )
        nodes = [node.op_type for node in onnx_program.model.graph]
        self.assertEqual(nodes.count("Barrier"), 2)
        onnx_program.save("barrier.onnx")


if __name__ == "__main__":
    unittest.main()

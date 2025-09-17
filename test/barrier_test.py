import unittest

import torch
from torch_onnx_models.components import _barrier


@_barrier.with_barrier
def decorated_func(x, y):
    return x * 2 + y


class Model(torch.nn.Module):
    def forward(self, x, y):
        return decorated_func(x, y)


class BarrierTest(unittest.TestCase):
    def test_export(self):
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

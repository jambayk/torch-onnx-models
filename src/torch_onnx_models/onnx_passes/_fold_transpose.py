from __future__ import annotations

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters


class FoldTransposePass(ir.passes.InPlacePass):
    """Fold transposed initializers."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for name, value in model.graph.initializers.items():
            # Only fold it if there is only one consumer for now
            if len(value.uses()) != 1:
                continue
            usage = next(iter(value.uses()))
            node = usage.node
            if node.op_type != "Transpose":
                continue
            assert len(node.inputs) == 1
            assert len(node.outputs) == 1

            assert value.const_value is not None
            initializer = value.const_value
            shape = initializer.shape

            perm = node.attributes.get_ints("perm", reversed(range(len(shape))))

            # Create a lazy transposed tensor
            torch_tensor = initializer.raw
            if isinstance(torch_tensor, torch.Tensor):

                def tensor_func(tensor=torch_tensor):
                    tensor = tensor.permute(*perm)
                    return tensor_adapters.TorchTensor(tensor, name=name)
            else:

                def tensor_func(tensor=initializer):
                    array = tensor.numpy()
                    return tensor_adapters.TorchTensor(
                        torch.from_numpy(array).permute(*perm), name=name
                    )

            value.const_value = ir.LazyTensor(
                tensor_func,
                dtype=initializer.dtype,
                shape=ir.Shape(shape),
                name=name,
            )

            modified = True
        return ir.passes.PassResult(model, modified=modified)

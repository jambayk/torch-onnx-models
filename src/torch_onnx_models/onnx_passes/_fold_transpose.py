from __future__ import annotations

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters


class FoldTransposePass(ir.passes.InPlacePass):
    """Fold transposed initializers."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        old_initializers = []
        new_initializers = []

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
                    return tensor_adapters.TorchTensor(tensor.permute(*perm), name=name)
            else:

                def tensor_func(tensor=initializer):
                    array = tensor.numpy()
                    return tensor_adapters.TorchTensor(
                        torch.from_numpy(array).permute(*perm), name=name
                    )

            new_shape = ir.Shape([shape[i] for i in perm])

            assert value.name is not None

            new_value = ir.val(
                # Keep the same name
                name=value.name,
                dtype=initializer.dtype,
                shape=new_shape,
                const_value=ir.LazyTensor(
                    tensor_func,
                    dtype=initializer.dtype,
                    shape=new_shape,
                    name=name,
                ),
            )

            # Replace the output of the transpose node with the new value
            ir.convenience.replace_all_uses_with(node.outputs[0], new_value)
            # Avoid modifying the dict while iterating
            old_initializers.append(value.name)
            new_initializers.append(new_value)
            # Remove the transpose node
            model.graph.remove(node, safe=True)

            modified = True

        for name in old_initializers:
            del model.graph.initializers[name]
        for value in new_initializers:
            model.graph.initializers.add(value)

        return ir.passes.PassResult(model, modified=modified)

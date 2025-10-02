from __future__ import annotations

import onnx_ir as ir
import torch



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

            perm_attr = node.attributes.get_ints("perm")

            assert value.const_value is not None

            # Create a lazy transposed tensor
            raw_tensor = value.const_value.raw
            if isinstance(raw_tensor, torch.Tensor):
                def tensor_func(tensor=raw_tensor):
                    tensor = tensor.transpose(*perm_attr)
                    return tensor_adapters.TorchTensor(tensor, name=name)

                ir_tensor = ir.LazyTensor(
                    tensor_func,
                    dtype=onnx_dtype,
                    shape=ir.Shape(tensor.shape),
                    name=name,
                )
            transposed_tensor = value.const_value

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir
import onnx_ir.convenience as ir_convenience
import onnx_ir.passes.common as common_passes

from torch_onnx_models.onnx_passes import _collect_opsets


def _assert_shapes_match(values1: Sequence[ir.Value], values2: Sequence[ir.Value]):
    if len(values1) != len(values2):
        raise ValueError("Number of values do not match")
    for v1, v2 in zip(values1, values2):
        if v1.type != v2.type:
            raise ValueError(f"Value types do not match: {v1.type} != {v2.type}")
        if v1.shape != v2.shape:
            raise ValueError(f"Value shapes do not match: {v1.shape} != {v2.shape}")


def replace_subgraph(
    model: ir.Model,
    inputs: Sequence[ir.Value],
    outputs: Sequence[ir.Value],
    replacement: ir.Graph,
    cleanup: bool = True,
) -> None:
    """Replace a subgraph defined by inputs and outputs with another graph.

    Args:
        model: The model that contains the graph to modify.
        inputs: The input values to the subgraph to replace.
        outputs: The output values from the subgraph to replace.
        replacement: The graph to insert in place of the subgraph.
        cleanup: Whether to remove the replaced nodes and reconcile conflicts after replacement.
    """
    if replacement.initializers:
        raise ValueError("Replacement graph cannot have initializers")

    _assert_shapes_match(inputs, replacement.inputs)
    _assert_shapes_match(outputs, replacement.outputs)

    replacement_inputs = list(replacement.inputs)
    replacement_outputs = list(replacement.outputs)
    new_nodes = list(replacement)

    # Remove nodes and values from the replacement
    replacement.inputs.clear()
    replacement.outputs.clear()
    replacement.initializers.clear()
    replacement.remove(new_nodes)

    insertion_point = outputs[0].producer()
    assert insertion_point is not None

    # First connect all new inputs to the original node outputs
    ir.convenience.replace_all_uses_with(replacement_inputs, inputs)

    # Then make all original consumers to use the new outputs
    ir_convenience.replace_nodes_and_values(
        model.graph,
        insertion_point,
        (),
        new_nodes,
        outputs,
        replacement_outputs,
    )

    if not cleanup:
        return

    # Clean up any nodes that are now dead and reconcile conflicts
    common_passes.RemoveUnusedNodesPass()(model)
    _collect_opsets.CollectOpsetsPass()(model)
    common_passes.RemoveUnusedOpsetsPass()(model)
    common_passes.NameFixPass()(model)

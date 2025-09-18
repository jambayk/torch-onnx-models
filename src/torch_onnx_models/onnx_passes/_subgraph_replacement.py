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
    graph: ir.Graph,
    inputs: Sequence[ir.Value],
    outputs: Sequence[ir.Value],
    replacement: ir.Graph,
    cleanup: bool = True,
) -> None:
    """Replace a subgraph defined by inputs and outputs with another graph.

    Args:
        graph: The graph to modify.
        inputs: The input values to the subgraph to replace.
        outputs: The output values from the subgraph to replace.
        replacement: The graph to insert in place of the subgraph.
        cleanup: Whether to run cleanup passes after replacement.
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
        graph,
        insertion_point,
        (),
        new_nodes,
        outputs,
        replacement_outputs,
    )

    if not cleanup:
        return

    # Clean up any nodes that are now dead and reconcile conflicts
    container_model = ir.Model(graph, ir_version=10)
    common_passes.RemoveUnusedNodesPass()(container_model)
    _collect_opsets.CollectOpsetsPass()(container_model)
    common_passes.RemoveUnusedOpsetsPass()(container_model)
    common_passes.NameFixPass()(container_model)

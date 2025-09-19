from __future__ import annotations

import onnx_ir as ir
import onnx_ir.convenience as ir_convenience


def _pop_node(node: ir.Node) -> None:
    """Remove a node from the graph, reconnecting its inputs to its outputs."""
    inputs = tuple(node.inputs)
    outputs = tuple(node.outputs)

    if len(inputs) != len(outputs):
        raise ValueError(
            f"Can only pop nodes with the same number of inputs and outputs: {node}"
        )

    ir_convenience.replace_all_uses_with(outputs, inputs)

    # Update graph outputs if the node generates output
    replacement_mapping = dict(zip(outputs, inputs))

    graph = node.graph
    assert graph is not None

    if any(out.is_graph_output() for out in node.outputs):
        for idx, graph_output in enumerate(graph.outputs):
            if graph_output in replacement_mapping:
                replacement_output = replacement_mapping[graph_output]
                assert replacement_output is not None
                graph.outputs[idx] = replacement_output

                # Maintain the name of the graph output
                replacement_output.name = graph_output.name

    # Finally remove the node
    graph.remove(node, safe=True)


class RemoveBarrierPass(ir.passes.InPlacePass):
    """Remove all Barrier nodes from the graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if node.op_type != "Barrier":
                continue
            _pop_node(node)
            modified = True

        return ir.passes.PassResult(model, modified=modified)

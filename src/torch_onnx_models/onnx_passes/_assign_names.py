"""Pass to assign names to onnx values based on model hierarchy."""

from __future__ import annotations

import ast
import logging
import typing

import onnx_ir as ir
from onnx_ir.passes._pass_infra import PassResult

logger = logging.getLogger(__name__)


class AssignNamesPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if "pkg.torch.onnx.name_scopes" in node.metadata_props:
                name_scopes = typing.cast(
                    "list[str]",
                    ast.literal_eval(node.metadata_props["pkg.torch.onnx.name_scopes"]),
                )
                name_scopes.pop()  # Remove self name
                for output in node.outputs:
                    if (
                        not output.is_graph_output()
                        and output.name is not None
                        and output.name != ""
                    ):
                        scoped_name = "/".join((*name_scopes, output.name))
                        logger.debug("Renaming %r to %r", output.name, scoped_name)
                        output.name = scoped_name
                        modified = True
        return PassResult(model, modified)

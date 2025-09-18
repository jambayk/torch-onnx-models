from __future__ import annotations

import onnx_ir as ir


def _maybe_set_opset_version(
    opset_imports: dict[str, int], domain: str, version: int | None
) -> bool:
    """Set the opset version for the domain."""
    if domain in opset_imports and opset_imports[domain] != 1:
        # Already set
        return False
    if version is None:
        # We don't know the opset version. Do nothing.
        return False
    # Set the known opset version for the domain
    opset_imports[domain] = version
    return True


class CollectOpsetsPass(ir.passes.InPlacePass):
    """Add opset imports for all used domains in the graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            domain = node.domain
            modified = (
                _maybe_set_opset_version(model.opset_imports, domain, node.version)
                or modified
            )

        # Functions are not handled in this pass

        return ir.passes.PassResult(model, modified=modified)

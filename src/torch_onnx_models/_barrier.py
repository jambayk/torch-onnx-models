from __future__ import annotations

import functools
import inspect
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import onnx_ir as ir
import torch

COUNTER = Counter()

TOptionalTensorSequence = TypeVar(
    "TOptionalTensorSequence",
    tuple[torch.Tensor, ...],
    torch.Tensor,
)


def barrier_op(
    inputs: TOptionalTensorSequence,
    attributes: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    region_identifier: str,
    type: Literal["input", "output"],
) -> TOptionalTensorSequence:
    """A custom ONNX operator that acts as a barrier for a subgraph.

    Args:
        inputs: A tuple of Tensors or a single Tensor. Some elements can be None.
        attributes: A dictionary of attributes for this barrier. These are user-defined
            and can be used to pass information about the subgraph.
        metadata: Metadata for this barrier / subgraph.
        region_identifier: A unique identifier for the barrier region. This should be
            the same for the input and output barriers of a subgraph.
        type: Either "input" or "output", indicating whether this is the input or output
            barrier of a subgraph.

    Returns:
        The same number of Tensors as inputs, in the same order. If an input is None,
        the corresponding output will also be None.
    """
    if not torch.onnx.is_in_onnx_export():
        # No-op outside of ONNX export
        return inputs

    # NOTE: inputs can have None values but that makes typing hard. So we don't annotate
    if metadata is None:
        metadata = {}

    if attributes is None:
        attributes = {}

    if isinstance(inputs, torch.Tensor) or inputs is None:
        tensors = (inputs,)
    else:
        tensors = inputs

    outputs: Sequence[torch.Tensor] = torch.onnx.ops.symbolic_multi_out(
        "pkg.olive::Barrier",
        tensors,
        attrs={
            "region_identifier": region_identifier,
            "type": type,
            "attributes": json.dumps(attributes),
            "metadata": json.dumps(metadata),
        },
        dtypes=[0 if t is None else t.dtype for t in tensors],
        shapes=[[] if t is None else t.shape for t in tensors],
        version=1,
    )
    result = [
        output if input is not None else None for output, input in zip(outputs, tensors)
    ]
    if len(result) == 1:
        return result[0]
    return result


def get_identifier(hint: str) -> str:
    if not hint:
        hint = "anonymous_region"
    count = COUNTER[hint]
    COUNTER[hint] += 1
    return f"{hint}_{count}"


def with_barrier(
    metadata: dict[str, Any] | None = None,
):
    """A decorator for inserting a pair of barriers for a subgraph.

    Args:
        func: A function that takes Tensors as positional arguments, other attributes
            as kwargs, and returns only one or more Optional[Tensor].
        metadata: Metadata for this pair of barrier / subgraph.
    """

    def decorator(func):
        signature = inspect.signature(func)
        default_values = {}
        for name, parameter in signature.parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                default_values[name] = parameter.default

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            identifier = get_identifier(func.__name__)
            attributes = default_values.copy()
            attributes.update(kwargs)
            inputs = barrier_op(
                args,
                attributes=attributes,
                metadata=metadata or {},
                region_identifier=identifier,
                type="input",
            )
            outputs = func(*inputs, **kwargs)
            return barrier_op(
                outputs,
                attributes={},
                metadata=metadata or {},
                region_identifier=identifier,
                type="output",
            )

        return wrapper

    return decorator


def get_attrs(node: ir.Node) -> dict[str, Any]:
    """Obtain the attributes dictionary from a Barrier node."""
    if node.op_type != "Barrier":
        raise ValueError(f"Node is not a Barrier: {node}")

    attrs_str = node.attributes.get_string("attributes")
    if not attrs_str:
        return {}
    return json.loads(attrs_str)

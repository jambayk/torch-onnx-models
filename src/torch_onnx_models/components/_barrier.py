from __future__ import annotations

import functools
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, Optional, TypeVar

import torch

COUNTER = Counter()

TOptionalTensorSequence = TypeVar(
    "TOptionalTensorSequence",
    tuple[torch.Tensor, ...],
    torch.Tensor,
)


def barrier_op(
    inputs: TOptionalTensorSequence,
    metadata: dict[str, Any] | None = None,
    *,
    group_identifier: str,
    type: Literal["input", "output"],
) -> TOptionalTensorSequence:
    # NOTE: inputs can have None values but that makes typing hard. So we don't annotate
    if metadata is None:
        metadata = {}

    if isinstance(inputs, torch.Tensor) or inputs is None:
        tensors = (inputs,)
    else:
        tensors = inputs

    outputs: Sequence[torch.Tensor] = torch.onnx.ops.symbolic_multi_out(
        "pkg.torch::Barrier",
        tensors,
        attrs={
            "group_identifier": group_identifier,
            "type": type,
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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            identifier = get_identifier(func.__name__)
            metadata_with_attrs = {"__attrs__": kwargs}
            metadata_with_attrs.update(metadata or {})
            inputs = barrier_op(
                args,
                metadata=metadata_with_attrs,
                group_identifier=identifier,
                type="input",
            )
            outputs = func(*inputs, **kwargs)
            return barrier_op(
                outputs,
                metadata=metadata_with_attrs,
                group_identifier=identifier,
                type="output",
            )

        return wrapper

    return decorator

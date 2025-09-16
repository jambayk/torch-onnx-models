from __future__ import annotations
from collections.abc import Sequence
import functools
import json
from typing import Literal

import torch


def barrier_op(
    inputs: Sequence[torch.Tensor | None],
    metadata: dict[
        str, bool | int | float | str | Sequence[int] | Sequence[float] | Sequence[str]
    ],
    group_identifier: str,
    type: Literal["input", "output"],
) -> Sequence[torch.Tensor | None]:
    outputs = torch.onnx.ops.symbolic_multi_out(
        "pkg.torch::Barrier",
        inputs,
        attrs={"group_identifier": group_identifier, "type": type},
        dtypes=[0 if t is None else t.dtype for t in inputs],
        shapes=[[] if t is None else t.shape for t in inputs],
        version=1,
        metadata_props={"metadata": json.dumps(metadata)},
    )
    return [
        output if input is not None else None for output, input in zip(outputs, inputs)
    ]


def _create_identifier(hint: str) -> str: ...


def with_barrier(
    func,
    metadata: dict[
        str, bool | int | float | str | Sequence[int] | Sequence[float] | Sequence[str]
    ],
):
    """A decorator for inserting a pair of barriers for a subgraph.

    Args:
        func: A function that takes Tensors as positional arguments, other attributes
            as kwargs, and returns only one or more Optional[Tensor].
        metadata: Metadata for this pair of barrier / subgraph.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        identifier = _create_identifier(func.__name__)
        inputs = barrier_op(
            args, metadata=metadata, group_identifier=identifier, type="input"
        )
        outputs = func(*inputs, **kwargs)
        return barrier_op(
            outputs, metadata=metadata, group_identifier=identifier, type="output"
        )

    return wrapper

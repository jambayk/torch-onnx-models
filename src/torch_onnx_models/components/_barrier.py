from __future__ import annotations

from collections.abc import Sequence
import functools
import json
from typing import Literal
import random
import string


import torch


def barrier_op(
    inputs: Sequence[torch.Tensor | None] | torch.Tensor,
    metadata: dict[
        str, bool | int | float | str | Sequence[int] | Sequence[float] | Sequence[str]
    ]
    | None = None,
    *,
    group_identifier: str,
    type: Literal["input", "output"],
) -> Sequence[torch.Tensor | None]:
    if metadata is None:
        metadata = {}

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    outputs = torch.onnx.ops.symbolic_multi_out(
        "pkg.torch::Barrier",
        inputs,
        attrs={
            "group_identifier": group_identifier,
            "type": type,
            "metadata": json.dumps(metadata),
        },
        dtypes=[0 if t is None else t.dtype for t in inputs],
        shapes=[[] if t is None else t.shape for t in inputs],
        version=1,
    )
    return [
        output if input is not None else None for output, input in zip(outputs, inputs)
    ]


def _create_identifier(hint: str) -> str:
    length = 8
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=length)
    )
    if not hint:
        hint = "anonymous_func"
    return f"{hint}_{random_string}"


def with_barrier(
    func,
    metadata: dict[
        str, bool | int | float | str | Sequence[int] | Sequence[float] | Sequence[str]
    ]
    | None = None,
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

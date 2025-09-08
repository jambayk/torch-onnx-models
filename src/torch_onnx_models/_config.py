from __future__ import annotations

import dataclasses
from typing import Literal

# Prototype of a config:

@dataclasses.dataclass
class Config:
    opset: Literal["onnx", "ort"] = "onnx"

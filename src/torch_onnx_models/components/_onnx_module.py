from __future__ import annotations

import abc

import torch
from torch import nn


class ONNXModule(nn.Module, abc.ABC):
    def forward(self, *args, **kwargs):
        if torch.onnx.is_in_onnx_export():
            return self.onnx_forward(*args, **kwargs)
        return self.reference(*args, **kwargs)

    @abc.abstractmethod
    def onnx_forward(self, *args, **kwargs):
        raise NotImplementedError("ONNX forward method not implemented.")

    @abc.abstractmethod
    def reference(self, *args, **kwargs):
        raise NotImplementedError("Reference method not implemented.")

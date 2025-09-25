from __future__ import annotations


import torch
from torch_onnx_models import _configs


def _create_example_inputs(config: _configs.ArchitectureConfig, export_config: _configs.ExportConfig):
    # Create example inputs and dynamic axes for ONNX export

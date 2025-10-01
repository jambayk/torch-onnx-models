from __future__ import annotations

__all__ = ["convert_hf_model"]

import json

import os
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_onnx_models import _configs, onnx_passes
from torch_onnx_models.models.llama.modeling_llama import LlamaForCausalLM


def _create_example_inputs(
    config: _configs.ArchitectureConfig, export_config: _configs.ExportConfig
):
    """Create example inputs and dynamic axes for ONNX export."""
    num_hidden_layers = config.num_hidden_layers
    batch = "batch"
    sequence_len = "sequence_len"
    past_sequence_len = "past_sequence_len"

    dynamic_shapes = {
        "input_ids": {0: batch, 1: sequence_len},
        "attention_mask": {
            0: batch,
            1: "past_sequence_len+sequence_len",
        },
        "position_ids": {
            0: batch,
            1: sequence_len,
        },
        "past_key_values": [
            ({0: batch, 2: past_sequence_len}, {0: batch, 2: past_sequence_len})
            for _ in range(num_hidden_layers)
        ],
    }

    example_batch_size = 2
    example_past_sequence_len = 2
    example_sequence_len = 3
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    example_inputs = dict(
        input_ids=torch.randint(
            0, 2, (example_batch_size, example_sequence_len), dtype=torch.int64
        ),
        attention_mask=torch.ones(
            (example_batch_size, example_past_sequence_len + example_sequence_len),
            dtype=torch.bool,
        ),
        position_ids=torch.arange(
            example_past_sequence_len,
            example_past_sequence_len + example_sequence_len,
            dtype=torch.int64,
        ).expand((example_batch_size, -1)),
        past_key_values=[
            (
                torch.randn(
                    example_batch_size,
                    num_key_value_heads,
                    example_sequence_len,
                    head_dim,
                ),
                torch.randn(
                    example_batch_size,
                    num_key_value_heads,
                    example_sequence_len,
                    head_dim,
                ),
            )
            for _ in range(num_hidden_layers)
        ],
    )

    return example_inputs, dynamic_shapes


def convert_hf_model(
    model_id: str = "meta-llama/Llama-2-7b-hf",
    load_weights: bool = True,
    clear_metadata: bool = False,
) -> torch.onnx.ONNXProgram:
    """Convert a HuggingFace model to ONNX.

    Args:
        model_id: The model ID on HuggingFace Hub.
        load_weights: Whether to load the pretrained weights from the HuggingFace model.
        clear_metadata: Whether to clear debugging metadata from the ONNX model.
    """
    from huggingface_hub import hf_hub_download

    # NOTE: No need to use transformers to load config
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    architecture_config = _configs.ArchitectureConfig.from_transformers(config)

    example_inputs, dynamic_shapes = _create_example_inputs(architecture_config, None)

    with FakeTensorMode():
        model = LlamaForCausalLM(architecture_config)

    onnx_program = torch.onnx.export(
        model,
        kwargs=example_inputs,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        optimize=False,
        opset_version=23,
    )

    assert onnx_program is not None

    onnx_program.model.producer_name = "torch_onnx_models"
    onnx_program.model.producer_version = "0.1.0"
    onnx_program.model.graph.name = model_id

    if load_weights:
        import safetensors.torch
        import onnx_safetensors

        # TODO: Support changing local_dir later
        safetensors_index_path = hf_hub_download(
            repo_id=model_id, filename="model.safetensors.index.json"
        )
        with open(safetensors_index_path) as f:
            safetensors_index = json.load(f)
        all_tensor_files = sorted(set(safetensors_index["weight_map"].values()))
        safetensors_paths = []
        print(f"Downloading {len(all_tensor_files)} safetensors files...")
        for tensor_file in all_tensor_files:
            # TODO(justinchuby): Concurrent download
            safetensors_paths.append(
                hf_hub_download(repo_id=model_id, filename=tensor_file)
            )
        initializers = {}
        for path in safetensors_paths:
            dir, filename = os.path.split(path)
            initializers.update(onnx_safetensors.read_safetensors(location=filename, base_dir=dir))

            # TODO(justinchuby): Validate missing keys
            # TODO(justinchuby): Handle dtype conversions
        onnx_safetensors.apply_tensors(onnx_program.model, initializers, cast=True)

    passes = ir.passes.PassManager(
        [
            onnx_passes.AssignNamesPass(),
            common_passes.DeduplicateInitializersPass(),
            common_passes.CommonSubexpressionEliminationPass(),
            common_passes.RemoveUnusedNodesPass(),
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
            # onnx_passes.RemoveBarrierPass()
        ]
    )

    passes(onnx_program.model)

    if clear_metadata:
        common_passes.ClearMetadataAndDocStringPass()(onnx_program.model)

    return onnx_program

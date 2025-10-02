from __future__ import annotations

__all__ = ["convert_hf_model"]

import json
import logging

import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import torch
from onnx_ir import tensor_adapters
from torch._subclasses.fake_tensor import FakeTensorMode
from onnxscript.optimizer import constant_folding

from torch_onnx_models import _configs, onnx_passes
from torch_onnx_models.components._model import CausalLMModel

logger = logging.getLogger(__name__)


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
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *[name for i in range(num_hidden_layers) for name in (f"past_key_values.{i}.key", f"past_key_values.{i}.value")],
    ]
    output_names = [
        "logits",
        *[name for i in range(num_hidden_layers) for name in (f"present.{i}.key", f"present.{i}.value")],
    ]

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
            dtype=torch.int64,
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

    return example_inputs, dynamic_shapes, input_names, output_names


def apply_weights(model: ir.Model, state_dict: dict[str, torch.Tensor]):
    """Apply weights from a state dict to an ONNX model."""
    for name, tensor in state_dict.items():
        if name not in model.graph.initializers:
            logger.warning(f"Weight '{name}' not found in the model. Skipped applying.")
            continue

        onnx_dtype = model.graph.initializers[name].dtype
        assert onnx_dtype is not None
        target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)
        if tensor.dtype != target_dtype:
            print(
                f"Converting weight '{name}' from {tensor.dtype} to {target_dtype}."
            )

            def tensor_func(tensor=tensor, target_dtype=target_dtype, name=name):
                tensor = tensor.to(target_dtype)
                return tensor_adapters.TorchTensor(tensor, name=name)

            ir_tensor = ir.LazyTensor(
                tensor_func,
                dtype=onnx_dtype,
                shape=ir.Shape(tensor.shape),
                name=name,
            )
        else:
            ir_tensor = tensor_adapters.TorchTensor(tensor, name)
        model.graph.initializers[name].const_value = ir_tensor


@torch.no_grad()
def convert_hf_model(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    load_weights: bool = True,
    clear_metadata: bool = False,
) -> torch.onnx.ONNXProgram:
    """Convert a HuggingFace model to ONNX.

    Args:
        model_id: The model ID on HuggingFace Hub.
        load_weights: Whether to load the pretrained weights from the HuggingFace model.
        clear_metadata: Whether to clear debugging metadata from the ONNX model.
    """
    import transformers

    # Need to use transformers to load config because transformers has additional
    # logic to standardize the config field names.
    config = transformers.AutoConfig.from_pretrained(model_id)
    architecture_config = _configs.ArchitectureConfig.from_transformers(config)

    example_inputs, dynamic_shapes, input_names, output_names = _create_example_inputs(architecture_config, None)

    with FakeTensorMode():
        model = CausalLMModel(architecture_config)

    onnx_program = torch.onnx.export(
        model,
        kwargs=example_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
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
        from huggingface_hub import hf_hub_download

        # TODO: Support changing local_dir later
        try:
            safetensors_index_path = hf_hub_download(
                repo_id=model_id, filename="model.safetensors.index.json"
            )
            with open(safetensors_index_path) as f:
                safetensors_index = json.load(f)
            all_tensor_files = sorted(set(safetensors_index["weight_map"].values()))
        except Exception as e:
            if "Entry Not Found" in str(e):
                # Fallback to single file
                all_tensor_files = ["model.safetensors"]
            else:
                raise e
        state_dict = {}
        safetensors_paths = []
        print(f"Downloading {len(all_tensor_files)} safetensors files...")
        for tensor_file in all_tensor_files:
            # TODO(justinchuby): Concurrent download
            safetensors_paths.append(
                hf_hub_download(repo_id=model_id, filename=tensor_file)
            )
        for path in safetensors_paths:
            state_dict.update(safetensors.torch.load_file(path))
            # TODO(justinchuby): Validate missing keys
        # can we make this better? at least not hardcode the weight names?
        if config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["model.embed_tokens.weight"] = state_dict[
                    "lm_head.weight"
                ]
            elif "model.embed_tokens.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict[
                    "model.embed_tokens.weight"
                ]

        apply_weights(onnx_program.model, state_dict)

    passes = ir.passes.PassManager(
        [
            onnx_passes.AssignNamesPass(),
            # needs to be applied early, otherwise deserialization fails
            constant_folding.FoldConstantsPass(
                shape_inference=False,
                input_size_limit=constant_folding.DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
                output_size_limit=constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT
            ),
            common_passes.RemoveUnusedNodesPass(),
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
            common_passes.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0),
            common_passes.DeduplicateInitializersPass(),
            common_passes.CommonSubexpressionEliminationPass(),
            # onnx_passes.RemoveBarrierPass()
        ]
    )

    passes(onnx_program.model)

    if clear_metadata:
        common_passes.ClearMetadataAndDocStringPass()(onnx_program.model)

    return onnx_program

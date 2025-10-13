from functools import partial

from transformers import AutoProcessor, AutoConfig

from torch_onnx_models._configs import ArchitectureConfig
from torch_onnx_models._exporter import convert_hf_model, create_text_gen_example_inputs
from torch_onnx_models.models import Gemma3ConditionalGenerationModel

models = {
    "gemma-3-4b-mm": "google/gemma-3-4b-it",
}


def create_gemma_example_inputs(config, model_id):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": (
                        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
                    ),
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    seq_len = inputs["input_ids"].shape[1]

    example_inputs, dynamic_shapes, input_names, output_names = create_text_gen_example_inputs(
        ArchitectureConfig.from_transformers(config.text_config), seq_len=seq_len
    )
    example_inputs["input_ids"] = inputs["input_ids"].repeat(2, 1)
    example_inputs["pixel_values"] = inputs["pixel_values"].repeat(2, 1, 1, 1)
    dynamic_shapes["pixel_values"] = {0: "batch"}
    input_names.append("pixel_values")

    return example_inputs, dynamic_shapes, input_names, output_names


for name, model_id in models.items():
    print(f"Exporting {model_id} to ONNX...")
    onnx_program = convert_hf_model(
        model_class=Gemma3ConditionalGenerationModel,
        config=AutoConfig.from_pretrained(model_id),
        example_inputs_func=partial(create_gemma_example_inputs, model_id=model_id),
        model_id=model_id,
        clear_metadata=True,
        # don't run CSE since we have to split the model cleanly later
        run_cse=False,
    )
    # TODO: Show progress bar
    print(f"Saving ONNX model to {name}.onnx...")
    onnx_program.save(f"{name}.onnx", external_data=True)
print("Done!")

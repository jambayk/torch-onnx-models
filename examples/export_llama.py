from torch_onnx_models._exporter import convert_hf_text_gen_model

models = {
    # "llama-3_2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    # "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    # "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-3-270m": "google/gemma-3-270m-it",
    # "gemma-3-1b": "google/gemma-3-1b-it",
    # "gemma-3-4b": "google/gemma-3-4b-it",
    # "phi-3_5-mini": "microsoft/Phi-3.5-mini-instruct",
    # "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    # "qwen-2_5-1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}

for name, model_id in models.items():
    print(f"Exporting {model_id} to ONNX...")
    onnx_program = convert_hf_text_gen_model(model_id, clear_metadata=True)
    # TODO: Show progress bar
    print(f"Saving ONNX model to {name}.onnx...")
    onnx_program.save(f"{name}.onnx", external_data=True)
print("Done!")

from torch_onnx_models._exporter import convert_hf_model

onnx_program = convert_hf_model("meta-llama/Llama-2-7b-hf", load_weights=True)
# TODO: Show progress bar
print("Saving ONNX model to llama-2-7b.onnx...")
onnx_program.save("llama-2-7b.onnx", external_data=True)
print("Done!")

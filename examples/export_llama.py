from torch_onnx_models._exporter import convert_hf_model

onnx_program = convert_hf_model("meta-llama/Llama-3.2-1B-Instruct", load_weights=True)
# TODO: Show progress bar
print("Saving ONNX model to llama-3_2-1b.onnx...")
onnx_program.save("llama-3_2-1b.onnx", external_data=True)
print("Done!")

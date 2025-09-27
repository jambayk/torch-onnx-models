from torch_onnx_models._exporter import convert_hf_model

onnx_program = convert_hf_model("meta-llama/Llama-2-7b-hf", load_weights=True)

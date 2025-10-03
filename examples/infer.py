# prerequisites
# clone and install olive https://github.com/microsoft/Olive
# install nightly ort-gpu
# pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple "onnxruntime-gpu==1.23.0.dev20251001001"
import sys

# add the generator script directory to the path (this is not part of the olive wheel)
sys.path.append("Olive/scripts")

from generator import ORTGenerator
from transformers import AutoTokenizer

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_path = "llama-3_2-1b.onnx"
# base_model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_path = "llama-2-7b.onnx"
# base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_path = "mistral-7b.onnx"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

# load the generator
generator = ORTGenerator(
    model_path, 
    tokenizer, 
    execution_provider="CUDAExecutionProvider",
    # execution_provider="CPUExecutionProvider",
    adapters={
        "base": {
            # llama 3
            "template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            # llama 2
            # "template": "{prompt}"
            # mistral
            # "template": "[INST]{prompt}[/INST]"
        }
    }
)

prompt = "Why is the sky blue?"
print(generator.generate(prompt, adapter="base", max_gen_len=100))

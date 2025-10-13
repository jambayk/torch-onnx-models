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
)

prompt = "Why is the sky blue? Can it you explain it to me like I'm five years old?"
full_prompt = (
    tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if tokenizer.chat_template is not None
    else prompt
)
print(generator.generate(full_prompt, max_gen_len=100))

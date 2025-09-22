# torch-onnx-models

A collection of PyTorch models and components optimized for ONNX export, featuring advanced techniques for model optimization and subgraph replacement.

## Overview

This project provides PyTorch neural network components specifically designed to be compatible with ONNX export workflows. It includes implementations of common transformer components like attention mechanisms, MLP layers, normalization, and activation functions, along with sophisticated ONNX graph manipulation tools.

## Background

Modern deep learning deployments often require converting PyTorch models to ONNX format for optimized inference across different platforms and hardware. However, standard PyTorch components don't always export cleanly to ONNX, especially for complex operations like attention mechanisms with rotary position embeddings (RoPE) or custom activation functions.

This project addresses these challenges by:

- **ONNX-First Design**: All components are built with ONNX export compatibility as a primary consideration
- **Barrier System**: A novel barrier mechanism that allows precise control over ONNX graph structure and enables targeted subgraph replacement
- **Optimized Components**: Implementations of transformer building blocks that export efficiently to ONNX
- **Graph Manipulation**: Tools for post-export ONNX graph optimization and customization

## Key Features

### 🧱 Neural Network Components
- **Attention**: Multi-head attention with support for different RoPE implementations
- **MLP**: Feed-forward network layers optimized for ONNX export
- **RMS Normalization**: Root Mean Square layer normalization
- **Activations**: Custom activation functions including QuickGELU

### 🚧 Barrier System
A unique approach to controlling ONNX graph structure through "barriers" that:
- Mark subgraph boundaries during export
- Enable precise subgraph replacement in post-processing
- Preserve metadata and attributes for optimization passes
- Support both decorator and functional API styles
- **Module-Level Barriers**: Automatic barrier insertion around entire PyTorch modules

### ⚖️ Weight Loading System
Comprehensive utilities for loading pre-trained weights:
- **HuggingFace Models**: Direct loading from HuggingFace transformers
- **PyTorch Checkpoints**: Support for .pt, .pth, and .bin files
- **SafeTensors**: Loading from SafeTensors format
- **Automatic Mapping**: Intelligent weight key mapping between different model formats
- **Flexible Configuration**: Customizable mappings for any model architecture

### 🏗️ Config Builder System
Automated configuration generation from popular model formats:
- **HuggingFace Integration**: Read HF configs and generate compatible configurations
- **Multi-Architecture Support**: Built-in support for LLaMA, Mistral, Phi, Gemma, Qwen
- **Model Factory**: Automatic component creation from configurations
- **Quick Utilities**: One-line functions for common use cases

### 🔧 ONNX Passes
Post-export optimization tools including:
- Subgraph replacement for custom operator injection
- Barrier removal for clean final graphs
- Opset collection and management
- Name assignment and conflict resolution

## Project Structure

```
src/torch_onnx_models/
├── components/           # Neural network building blocks
│   ├── _attention.py    # Multi-head attention implementation
│   ├── _mlp.py         # Feed-forward network layers
│   ├── _rms_norm.py    # RMS normalization
│   └── _activations.py # Custom activation functions
├── onnx_passes/         # ONNX graph optimization tools
│   ├── _subgraph_replacement.py  # Replace subgraphs post-export
│   ├── _barrier_removal.py       # Remove barrier annotations
│   └── _collect_opsets.py        # Manage ONNX opsets
├── _barrier.py          # Core barrier system implementation
├── _module_barriers.py  # Module-level barrier system
├── _weight_loading.py   # Weight loading utilities
└── _config_builder.py   # HuggingFace config conversion system
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.8.0+
- ONNX
- ONNXScript

### Install from Source

1. Clone the repository:
```cmd
git clone https://github.com/leestott/torch-onnx-models.git
cd torch-onnx-models
```

2. Install in development mode:
```cmd
pip install -e .
```

3. Install with optional dependencies for enhanced features:
```cmd
# For HuggingFace integration
pip install -e ".[huggingface]"

# For SafeTensors support
pip install -e ".[safetensors]"

# For all optional features
pip install -e ".[full]"
```

## Usage

### Basic Component Usage

```python
import torch
from argparse import Namespace
from torch_onnx_models.components import Attention
from torch_onnx_models.components._attention_utils import create_attention_bias

# Configure attention layer
config = Namespace(
    hidden_size=2048,
    head_dim=64,
    num_attention_heads=32,
    num_key_value_heads=8,
    attention_bias=False
)

# Create attention module
attention = Attention(config)

# Prepare inputs
batch_size, seq_length = 1, 10
hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
# ... other inputs ...

# Export to ONNX
onnx_program = torch.onnx.export(
    attention,
    (hidden_states, attention_bias, position_ids, cos_cache, sin_cache, past_key, past_value),
    opset_version=23,
    dynamo=True
)
```

### Using the Barrier System

#### Decorator Approach
```python
from torch_onnx_models._barrier import with_barrier

@with_barrier(metadata={"region": "custom_op"})
def my_custom_function(x, y):
    return x * 2 + y

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return my_custom_function(x, y)
```

#### Functional Approach
```python
from torch_onnx_models._barrier import barrier_op, get_identifier

def my_model_forward(x, y):
    identifier = get_identifier("my_region")
    
    # Input barrier
    x, y = barrier_op(
        (x, y),
        attributes={"operation": "custom"},
        region_identifier=identifier,
        type="input"
    )
    
    # Your computation
    result = x * 2 + y
    
    # Output barrier
    result = barrier_op(
        (result,),
        region_identifier=identifier,
        type="output"
    )
    return result
```

### Post-Export Graph Manipulation

```python
import onnx_ir as ir
from torch_onnx_models.onnx_passes._subgraph_replacement import replace_subgraph

# Load your exported model
model = ir.from_proto(onnx_model)

# Define replacement graph for a specific subgraph
# ... create replacement_graph ...

# Replace subgraph between barriers
replace_subgraph(
    model=model,
    inputs=subgraph_inputs,
    outputs=subgraph_outputs,
    replacement=replacement_graph,
    cleanup=True
)
```

### Weight Loading

#### Loading from HuggingFace Models
```python
from torch_onnx_models._weight_loading import WeightLoader
from torch_onnx_models.components._attention import Attention
from torch_onnx_models._config_builder import quick_config_from_hf

# Create model with HF config
config = quick_config_from_hf("microsoft/DialoGPT-medium")
attention = Attention(config.get_attention_config().to_namespace())

# Load weights
loader = WeightLoader()
stats = loader.load_from_huggingface(attention, "microsoft/DialoGPT-medium")
print(f"Loaded {stats['total_loaded']} parameters")
```

#### Loading from PyTorch Checkpoints
```python
from torch_onnx_models._weight_loading import load_weights_with_auto_mapping

# Automatic weight loading with mapping detection
stats = load_weights_with_auto_mapping(
    model=my_model,
    source_path="model_checkpoint.pt",
    ignore_missing=True
)
```

#### Custom Weight Mappings
```python
from torch_onnx_models._weight_loading import WeightLoader

loader = WeightLoader()
# Add custom mappings for non-standard architectures
loader.add_weight_mappings({
    "transformer.layers.0.attention.query.weight": "layers.0.attention.q_proj.weight",
    "transformer.layers.0.attention.key.weight": "layers.0.attention.k_proj.weight",
    # ... more mappings
})

stats = loader.load_from_checkpoint(model, "custom_model.pt")
```

### Module-Level Barriers

#### Automatic Barrier Application
```python
from torch_onnx_models._module_barriers import AutoBarrierModel, apply_barriers_to_model

# Wrap entire model with auto-barriers
barrier_model = AutoBarrierModel(
    model=my_transformer,
    auto_barrier_types=["Attention", "MLP", "LayerNorm"],
    barrier_metadata={"optimization_target": "inference"}
)

# Or apply to existing model
apply_barriers_to_model(
    my_model,
    target_modules=["Attention", "MLP"],
    metadata_fn=lambda name, module: {"module_name": name}
)
```

#### Module Decorator Approach
```python
from torch_onnx_models._module_barriers import with_module_barrier

@with_module_barrier(
    region_name="custom_attention",
    metadata={"optimization": "fused_attention"}
)
class MyAttention(nn.Module):
    def forward(self, x):
        # Your attention implementation
        return x
```

#### Targeted Module Wrapping
```python
from torch_onnx_models._module_barriers import wrap_attention_modules, wrap_mlp_modules

# Wrap only attention modules
model_with_attention_barriers = wrap_attention_modules(my_model)

# Wrap only MLP modules
model_with_mlp_barriers = wrap_mlp_modules(my_model)
```

### Config Builder System

#### Quick Model Creation from HuggingFace
```python
from torch_onnx_models._config_builder import (
    quick_config_from_hf,
    quick_attention_from_hf,
    quick_mlp_from_hf
)

# Get configuration
config = quick_config_from_hf("meta-llama/Llama-2-7b-hf")
print(f"Model has {config.num_hidden_layers} layers")

# Create components directly
attention = quick_attention_from_hf("meta-llama/Llama-2-7b-hf")
mlp = quick_mlp_from_hf("meta-llama/Llama-2-7b-hf")
```

#### Manual Config Building
```python
from torch_onnx_models._config_builder import ConfigBuilder, ModelFactory

builder = ConfigBuilder()
factory = ModelFactory()

# Build from local config file
config = builder.from_huggingface_config("./my_model/config.json")

# Create components
attention = factory.create_attention(config)
mlp = factory.create_mlp(config)
```

#### Custom Model Type Registration
```python
from torch_onnx_models._config_builder import config_builder, TransformerConfig

def build_my_custom_config(hf_config):
    return TransformerConfig(
        hidden_size=hf_config["d_model"],
        num_attention_heads=hf_config["n_heads"],
        # ... custom mappings
    )

# Register custom model type
config_builder.register_model_type("my_custom_model", build_my_custom_config)

# Now can build configs for your custom model type
config = config_builder.from_huggingface_config(my_config_dict, model_type="my_custom_model")
```

### Complete Workflow Example

```python
from torch_onnx_models._config_builder import quick_config_from_hf
from torch_onnx_models._weight_loading import load_weights_with_auto_mapping
from torch_onnx_models._module_barriers import AutoBarrierModel
from torch_onnx_models.components._attention import Attention

# 1. Create model from HuggingFace config
config = quick_config_from_hf("microsoft/DialoGPT-medium")
attention = Attention(config.get_attention_config().to_namespace())

# 2. Load pre-trained weights
weight_stats = load_weights_with_auto_mapping(
    model=attention,
    source_path="microsoft/DialoGPT-medium",
    ignore_missing=True
)

# 3. Add barriers for ONNX optimization
barrier_attention = AutoBarrierModel(
    model=attention,
    auto_barrier_types=["Attention"],
    barrier_metadata={"target": "onnx_export"}
)

# 4. Export to ONNX
inputs = (hidden_states, attention_bias, position_ids, cos_cache, sin_cache, past_key, past_value)
onnx_program = torch.onnx.export(
    barrier_attention,
    inputs,
    opset_version=23,
    dynamo=True
)

print(f"Successfully exported model with {weight_stats['total_loaded']} loaded parameters")
```

## Running Tests

The project includes comprehensive tests for all components:

```cmd
# Run all tests
python -m pytest test/

# Run specific test files
python -m pytest test/attention_test.py
python -m pytest test/barrier_test.py
python -m pytest test/onnx_passes_test.py
python -m pytest test/rms_norm_test.py
python -m pytest test/weight_loading_test.py
python -m pytest test/module_barriers_test.py
python -m pytest test/config_builder_test.py

# Run with verbose output
python -m pytest test/ -v
```

### Test Examples

- **Attention Test**: Validates ONNX export of attention mechanisms with different configurations
- **Barrier Test**: Demonstrates barrier usage and verifies correct ONNX graph structure
- **ONNX Passes Test**: Tests graph manipulation and optimization passes
- **RMS Norm Test**: Validates normalization layer export
- **Weight Loading Test**: Tests weight loading from various sources and formats
- **Module Barriers Test**: Validates module-level barrier functionality and auto-wrapping
- **Config Builder Test**: Tests HuggingFace config conversion and model factory

## Development

### Project Structure
This project follows a modular design:

- **Components**: Reusable neural network building blocks
- **ONNX Passes**: Post-export graph optimization tools  
- **Barrier System**: Core infrastructure for graph control
- **Tests**: Comprehensive validation of all functionality

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- **Jambay Kinley**
- **Justin Chu** 
- **Lee Stott**
- **Kinfey Lo**

## Related Projects

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [ONNX](https://onnx.ai/) - Open Neural Network Exchange
- [ONNXScript](https://github.com/microsoft/onnxscript) - ONNX graph construction and manipulation

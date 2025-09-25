"""Config builder system for torch-onnx-models.

This module provides utilities for reading HuggingFace model configurations
and automatically generating compatible model configurations and instances
for torch-onnx-models components.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn


class ModelConfig:
    """Base configuration class for torch-onnx-models."""

    def __init__(self, **kwargs):
        """Initialize configuration with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def to_namespace(self) -> Namespace:
        """Convert configuration to argparse.Namespace."""
        return Namespace(**self.to_dict())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class AttentionConfig(ModelConfig):
    """Configuration for attention modules."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize attention configuration."""
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling


class MLPConfig(ModelConfig):
    """Configuration for MLP modules."""

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        """Initialize MLP configuration."""
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.hidden_act = hidden_act
        self.bias = bias
        self.dropout = dropout


class TransformerConfig(ModelConfig):
    """Configuration for transformer models."""

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "gelu",
        attention_bias: bool = True,
        mlp_bias: bool = True,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        num_key_value_heads: Optional[int] = None,
        tie_word_embeddings: bool = False,
        **kwargs
    ):
        """Initialize transformer configuration."""
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.tie_word_embeddings = tie_word_embeddings

    def get_attention_config(self) -> AttentionConfig:
        """Extract attention configuration."""
        return AttentionConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attention_bias=self.attention_bias,
            max_position_embeddings=self.max_position_embeddings,
            rope_base=self.rope_base,
        )

    def get_mlp_config(self) -> MLPConfig:
        """Extract MLP configuration."""
        return MLPConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            bias=self.mlp_bias,
        )


class ConfigBuilder:
    """Builder for converting HuggingFace configs to torch-onnx-models configs."""

    def __init__(self):
        """Initialize the config builder."""
        self._model_type_mappings = {
            "llama": self._build_llama_config,
            "mistral": self._build_mistral_config,
            "phi": self._build_phi_config,
            "gemma": self._build_gemma_config,
            "qwen": self._build_qwen_config,
        }

    def register_model_type(self, model_type: str, builder_fn: callable) -> None:
        """Register a custom model type builder.
        
        Args:
            model_type: The model type identifier.
            builder_fn: Function that takes HF config dict and returns ModelConfig.
        """
        self._model_type_mappings[model_type] = builder_fn

    def from_huggingface_config(
        self,
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        model_type: Optional[str] = None,
    ) -> TransformerConfig:
        """Build config from HuggingFace configuration.
        
        Args:
            config_path_or_dict: Path to config.json or config dictionary.
            model_type: Override model type detection.
            
        Returns:
            Compatible TransformerConfig instance.
        """
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, 'r') as f:
                hf_config = json.load(f)
        else:
            hf_config = config_path_or_dict

        # Detect model type if not provided
        if model_type is None:
            model_type = hf_config.get("model_type", "").lower()

        # Use specific builder if available
        if model_type in self._model_type_mappings:
            return self._model_type_mappings[model_type](hf_config)
        else:
            # Fallback to generic builder
            return self._build_generic_config(hf_config)

    def from_huggingface_model(self, model_name_or_path: str) -> TransformerConfig:
        """Build config from HuggingFace model name or path.
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            
        Returns:
            Compatible TransformerConfig instance.
        """
        try:
            from transformers import AutoConfig
        except ImportError:
            raise ImportError(
                "transformers library is required for HuggingFace config loading. "
                "Install with: pip install transformers"
            )

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        return self.from_huggingface_config(hf_config.to_dict())

    def _build_llama_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build config for LLaMA-style models."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hf_config.get("hidden_size", 4096),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            intermediate_size=hf_config.get("intermediate_size", 11008),
            hidden_act=hf_config.get("hidden_act", "silu"),
            attention_bias=False,  # LLaMA doesn't use attention bias
            mlp_bias=False,       # LLaMA doesn't use MLP bias
            layer_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
            rope_base=hf_config.get("rope_theta", 10000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )

    def _build_mistral_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build config for Mistral-style models."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hf_config.get("hidden_size", 4096),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            intermediate_size=hf_config.get("intermediate_size", 14336),
            hidden_act=hf_config.get("hidden_act", "silu"),
            attention_bias=False,
            mlp_bias=False,
            layer_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
            max_position_embeddings=hf_config.get("max_position_embeddings", 32768),
            rope_base=hf_config.get("rope_theta", 10000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )

    def _build_phi_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build config for Phi-style models."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 51200),
            hidden_size=hf_config.get("hidden_size", 2048),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_hidden_layers=hf_config.get("num_hidden_layers", 24),
            intermediate_size=hf_config.get("intermediate_size", 8192),
            hidden_act=hf_config.get("hidden_act", "gelu_new"),
            attention_bias=hf_config.get("attention_bias", True),
            mlp_bias=hf_config.get("mlp_bias", True),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
            rope_base=hf_config.get("rope_theta", 10000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )

    def _build_gemma_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build config for Gemma-style models."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 256000),
            hidden_size=hf_config.get("hidden_size", 3072),
            num_attention_heads=hf_config.get("num_attention_heads", 24),
            num_hidden_layers=hf_config.get("num_hidden_layers", 28),
            intermediate_size=hf_config.get("intermediate_size", 24576),
            hidden_act=hf_config.get("hidden_activation", "gelu"),
            attention_bias=False,
            mlp_bias=False,
            layer_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            max_position_embeddings=hf_config.get("max_position_embeddings", 8192),
            rope_base=hf_config.get("rope_theta", 10000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
        )

    def _build_qwen_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build config for Qwen-style models."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 151936),
            hidden_size=hf_config.get("hidden_size", 4096),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            intermediate_size=hf_config.get("intermediate_size", 22016),
            hidden_act=hf_config.get("hidden_act", "silu"),
            attention_bias=hf_config.get("attention_bias", True),
            mlp_bias=False,
            layer_norm_eps=hf_config.get("layer_norm_epsilon", 1e-6),
            max_position_embeddings=hf_config.get("max_position_embeddings", 32768),
            rope_base=hf_config.get("rope_theta", 1000000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )

    def _build_generic_config(self, hf_config: Dict[str, Any]) -> TransformerConfig:
        """Build generic config for unknown model types."""
        return TransformerConfig(
            vocab_size=hf_config.get("vocab_size", 50257),
            hidden_size=hf_config.get("hidden_size", 768),
            num_attention_heads=hf_config.get("num_attention_heads", 12),
            num_hidden_layers=hf_config.get("num_hidden_layers", 12),
            intermediate_size=hf_config.get("intermediate_size"),
            hidden_act=hf_config.get("hidden_act", "gelu"),
            attention_bias=hf_config.get("attention_bias", True),
            mlp_bias=hf_config.get("mlp_bias", True),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            max_position_embeddings=hf_config.get("max_position_embeddings", 1024),
            rope_base=hf_config.get("rope_theta", 10000.0),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
        )


class ModelFactory:
    """Factory for creating torch-onnx-models components from configurations."""

    def __init__(self):
        """Initialize the model factory."""
        pass

    def create_attention(self, config: Union[AttentionConfig, TransformerConfig]) -> nn.Module:
        """Create an attention module from configuration."""
        from torch_onnx_models.components._attention import Attention
        
        if isinstance(config, TransformerConfig):
            config = config.get_attention_config()
        
        return Attention(config.to_namespace())

    def create_mlp(self, config: Union[MLPConfig, TransformerConfig]) -> nn.Module:
        """Create an MLP module from configuration."""
        from torch_onnx_models.components._mlp import MLP
        
        if isinstance(config, TransformerConfig):
            config = config.get_mlp_config()
        
        return MLP(config.to_namespace())

    def create_model_from_config(
        self,
        config: TransformerConfig,
        model_class: Optional[Type[nn.Module]] = None,
    ) -> nn.Module:
        """Create a complete model from configuration.
        
        Args:
            config: Model configuration.
            model_class: Optional custom model class to instantiate.
            
        Returns:
            Instantiated model.
        """
        if model_class is not None:
            return model_class(config)
        
        # For now, return a simple container with the config
        # Users can extend this to create actual model instances
        class ConfiguredModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
        return ConfiguredModel(config)


# Global instances
config_builder = ConfigBuilder()
model_factory = ModelFactory()


def quick_config_from_hf(model_name_or_path: str) -> TransformerConfig:
    """Quick utility to create config from HuggingFace model."""
    return config_builder.from_huggingface_model(model_name_or_path)


def quick_attention_from_hf(model_name_or_path: str) -> nn.Module:
    """Quick utility to create attention module from HuggingFace model."""
    config = quick_config_from_hf(model_name_or_path)
    return model_factory.create_attention(config)


def quick_mlp_from_hf(model_name_or_path: str) -> nn.Module:
    """Quick utility to create MLP module from HuggingFace model."""
    config = quick_config_from_hf(model_name_or_path)
    return model_factory.create_mlp(config)
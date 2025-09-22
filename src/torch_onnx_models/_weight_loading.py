"""Weight loading utilities for torch-onnx-models.

This module provides utilities for loading pre-trained weights from various sources
including HuggingFace transformers, PyTorch checkpoints, and other common formats.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import torch
import torch.nn as nn


class WeightLoader:
    """Utility class for loading and mapping weights to torch-onnx-models components."""

    def __init__(self, strict: bool = True):
        """Initialize the weight loader.
        
        Args:
            strict: Whether to strictly enforce weight shape matching.
        """
        self.strict = strict
        self._weight_mappings: Dict[str, str] = {}

    def add_weight_mapping(self, source_key: str, target_key: str) -> None:
        """Add a mapping from source weight key to target weight key.
        
        Args:
            source_key: Key in the source checkpoint.
            target_key: Key in the target model.
        """
        self._weight_mappings[source_key] = target_key

    def add_weight_mappings(self, mappings: Dict[str, str]) -> None:
        """Add multiple weight mappings.
        
        Args:
            mappings: Dictionary of source_key -> target_key mappings.
        """
        self._weight_mappings.update(mappings)

    def load_from_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        prefix: str = "",
        ignore_missing: bool = False,
    ) -> Dict[str, Any]:
        """Load weights from a PyTorch checkpoint.
        
        Args:
            model: The target model to load weights into.
            checkpoint_path: Path to the checkpoint file.
            prefix: Prefix to add to all keys when loading.
            ignore_missing: Whether to ignore missing keys in the checkpoint.
            
        Returns:
            Dictionary with loading statistics.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        return self._load_state_dict(model, state_dict, prefix, ignore_missing)

    def load_from_huggingface(
        self,
        model: nn.Module,
        model_name_or_path: str,
        ignore_missing: bool = False,
    ) -> Dict[str, Any]:
        """Load weights from a HuggingFace model.
        
        Args:
            model: The target model to load weights into.
            model_name_or_path: HuggingFace model name or local path.
            ignore_missing: Whether to ignore missing keys.
            
        Returns:
            Dictionary with loading statistics.
        """
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers library is required for HuggingFace weight loading. "
                "Install with: pip install transformers"
            )

        # Load the HuggingFace model
        hf_model = AutoModel.from_pretrained(model_name_or_path)
        hf_state_dict = hf_model.state_dict()

        return self._load_state_dict(model, hf_state_dict, "", ignore_missing)

    def load_from_safetensors(
        self,
        model: nn.Module,
        safetensors_path: Union[str, Path],
        prefix: str = "",
        ignore_missing: bool = False,
    ) -> Dict[str, Any]:
        """Load weights from a SafeTensors file.
        
        Args:
            model: The target model to load weights into.
            safetensors_path: Path to the SafeTensors file.
            prefix: Prefix to add to all keys when loading.
            ignore_missing: Whether to ignore missing keys.
            
        Returns:
            Dictionary with loading statistics.
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors library is required for SafeTensors loading. "
                "Install with: pip install safetensors"
            )

        safetensors_path = Path(safetensors_path)
        if not safetensors_path.exists():
            raise FileNotFoundError(f"SafeTensors file not found: {safetensors_path}")

        state_dict = load_file(str(safetensors_path))
        return self._load_state_dict(model, state_dict, prefix, ignore_missing)

    def _load_state_dict(
        self,
        model: nn.Module,
        source_state_dict: Dict[str, torch.Tensor],
        prefix: str,
        ignore_missing: bool,
    ) -> Dict[str, Any]:
        """Internal method to load state dict with mappings."""
        model_state_dict = model.state_dict()
        
        # Apply prefix if specified
        if prefix:
            source_state_dict = {
                f"{prefix}.{k}": v for k, v in source_state_dict.items()
            }

        # Apply weight mappings
        mapped_state_dict = {}
        for source_key, tensor in source_state_dict.items():
            target_key = self._weight_mappings.get(source_key, source_key)
            mapped_state_dict[target_key] = tensor

        # Filter to only include keys that exist in the model
        filtered_state_dict = {}
        missing_keys = []
        unexpected_keys = []

        for key in model_state_dict.keys():
            if key in mapped_state_dict:
                source_tensor = mapped_state_dict[key]
                target_tensor = model_state_dict[key]
                
                if source_tensor.shape != target_tensor.shape:
                    if self.strict:
                        raise ValueError(
                            f"Shape mismatch for key '{key}': "
                            f"source {source_tensor.shape} vs target {target_tensor.shape}"
                        )
                    else:
                        print(f"Warning: Shape mismatch for key '{key}' - skipping")
                        missing_keys.append(key)
                        continue
                
                filtered_state_dict[key] = source_tensor
            else:
                missing_keys.append(key)

        for key in mapped_state_dict.keys():
            if key not in model_state_dict:
                unexpected_keys.append(key)

        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)

        return {
            "loaded_keys": list(filtered_state_dict.keys()),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "total_loaded": len(filtered_state_dict),
            "total_missing": len(missing_keys),
            "total_unexpected": len(unexpected_keys),
        }


def create_common_weight_mappings() -> Dict[str, str]:
    """Create common weight mappings for popular model architectures.
    
    Returns:
        Dictionary of common weight mappings.
    """
    mappings = {}
    
    # Common HuggingFace -> torch-onnx-models mappings
    # Attention mappings
    mappings.update({
        "self_attn.q_proj.weight": "q_proj.weight",
        "self_attn.k_proj.weight": "k_proj.weight", 
        "self_attn.v_proj.weight": "v_proj.weight",
        "self_attn.o_proj.weight": "o_proj.weight",
        "attention.query.weight": "q_proj.weight",
        "attention.key.weight": "k_proj.weight",
        "attention.value.weight": "v_proj.weight",
        "attention.dense.weight": "o_proj.weight",
    })
    
    # MLP mappings
    mappings.update({
        "mlp.gate_proj.weight": "gate_proj.weight",
        "mlp.up_proj.weight": "up_proj.weight", 
        "mlp.down_proj.weight": "down_proj.weight",
        "feed_forward.w1.weight": "gate_proj.weight",
        "feed_forward.w2.weight": "down_proj.weight",
        "feed_forward.w3.weight": "up_proj.weight",
    })
    
    # Normalization mappings
    mappings.update({
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "norm.weight": "norm.weight",
        "layer_norm.weight": "layer_norm.weight",
    })
    
    return mappings


def load_weights_with_auto_mapping(
    model: nn.Module,
    source_path: Union[str, Path],
    model_type: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Load weights with automatic mapping detection.
    
    Args:
        model: Target model to load weights into.
        source_path: Path to weights (checkpoint, HF model, etc.).
        model_type: Optional model type hint for mapping selection.
        **kwargs: Additional arguments passed to the loader.
        
    Returns:
        Dictionary with loading statistics.
    """
    loader = WeightLoader(strict=kwargs.get("strict", True))
    
    # Add common mappings
    common_mappings = create_common_weight_mappings()
    loader.add_weight_mappings(common_mappings)
    
    # Detect file type and load accordingly
    source_path = Path(source_path)
    
    if source_path.is_dir():
        # Assume HuggingFace model directory
        return loader.load_from_huggingface(model, str(source_path), **kwargs)
    elif source_path.suffix == ".safetensors":
        return loader.load_from_safetensors(model, source_path, **kwargs)
    elif source_path.suffix in [".pt", ".pth", ".bin"]:
        return loader.load_from_checkpoint(model, source_path, **kwargs)
    else:
        # Try as HuggingFace model name
        return loader.load_from_huggingface(model, str(source_path), **kwargs)


class WeightMapper:
    """Utility for creating and managing weight mappings between different model formats."""
    
    def __init__(self):
        self.mappings: Dict[str, Dict[str, str]] = {}
    
    def add_mapping_set(self, name: str, mappings: Dict[str, str]) -> None:
        """Add a named set of weight mappings."""
        self.mappings[name] = mappings
    
    def get_mapping_set(self, name: str) -> Dict[str, str]:
        """Get a named set of weight mappings."""
        return self.mappings.get(name, {})
    
    def create_llama_mappings(self) -> Dict[str, str]:
        """Create weight mappings for LLaMA-style models."""
        return {
            # Embeddings
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            
            # Layer mappings (will need layer index replacement)
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.k_proj.weight", 
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.v_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.o_proj.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
        }
    
    def expand_layer_mappings(self, mappings: Dict[str, str], num_layers: int) -> Dict[str, str]:
        """Expand layer mappings with actual layer indices."""
        expanded = {}
        for source_pattern, target_pattern in mappings.items():
            if "{}" in source_pattern:
                for i in range(num_layers):
                    source_key = source_pattern.format(i)
                    target_key = target_pattern.format(i)
                    expanded[source_key] = target_key
            else:
                expanded[source_pattern] = target_pattern
        return expanded


# Global weight mapper instance
weight_mapper = WeightMapper()

# Add common mapping sets
weight_mapper.add_mapping_set("llama", weight_mapper.create_llama_mappings())
weight_mapper.add_mapping_set("common", create_common_weight_mappings())
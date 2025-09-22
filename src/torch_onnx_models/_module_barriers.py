"""Module-level barrier system for torch-onnx-models.

This module extends the barrier system to work at the PyTorch module level,
allowing automatic barrier insertion around entire modules during ONNX export.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn

from torch_onnx_models._barrier import barrier_op, get_identifier


class BarrierModule(nn.Module):
    """A wrapper module that automatically adds barriers around its forward pass."""

    def __init__(
        self,
        wrapped_module: nn.Module,
        region_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a barrier-wrapped module.
        
        Args:
            wrapped_module: The module to wrap with barriers.
            region_name: Custom name for the barrier region. If None, uses module class name.
            metadata: Metadata to attach to the barriers.
            attributes: Attributes to attach to the input barrier.
        """
        super().__init__()
        self.wrapped_module = wrapped_module
        self.region_name = region_name or wrapped_module.__class__.__name__.lower()
        self.metadata = metadata or {}
        self.attributes = attributes or {}
        self._region_identifier = None

    def forward(self, *args, **kwargs):
        """Forward pass with automatic barrier insertion."""
        if self._region_identifier is None:
            self._region_identifier = get_identifier(self.region_name)

        # Input barrier
        barrier_args = barrier_op(
            args,
            attributes=self.attributes,
            metadata=self.metadata,
            region_identifier=self._region_identifier,
            type="input",
        )

        # Forward pass through wrapped module
        outputs = self.wrapped_module(*barrier_args, **kwargs)

        # Output barrier
        barrier_outputs = barrier_op(
            outputs,
            attributes={},
            metadata=self.metadata,
            region_identifier=self._region_identifier,
            type="output",
        )

        return barrier_outputs

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_module, name)


def with_module_barrier(
    region_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to automatically add barriers around a module's forward pass.
    
    Args:
        region_name: Custom name for the barrier region.
        metadata: Metadata to attach to the barriers.
        attributes: Attributes to attach to the input barrier.
    """
    def decorator(module_class: Type[nn.Module]) -> Type[nn.Module]:
        original_forward = module_class.forward

        @functools.wraps(original_forward)
        def barrier_forward(self, *args, **kwargs):
            region_name_final = region_name or module_class.__name__.lower()
            identifier = get_identifier(region_name_final)

            # Input barrier
            barrier_args = barrier_op(
                args,
                attributes=attributes or {},
                metadata=metadata or {},
                region_identifier=identifier,
                type="input",
            )

            # Original forward pass
            outputs = original_forward(self, *barrier_args, **kwargs)

            # Output barrier
            barrier_outputs = barrier_op(
                outputs,
                attributes={},
                metadata=metadata or {},
                region_identifier=identifier,
                type="output",
            )

            return barrier_outputs

        module_class.forward = barrier_forward
        return module_class

    return decorator


def wrap_module_with_barrier(
    module: nn.Module,
    region_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> BarrierModule:
    """Wrap an existing module with barriers.
    
    Args:
        module: The module to wrap.
        region_name: Custom name for the barrier region.
        metadata: Metadata to attach to the barriers.
        attributes: Attributes to attach to the input barrier.
        
    Returns:
        A BarrierModule wrapping the original module.
    """
    return BarrierModule(
        wrapped_module=module,
        region_name=region_name,
        metadata=metadata,
        attributes=attributes,
    )


def apply_barriers_to_model(
    model: nn.Module,
    target_modules: Optional[list[str]] = None,
    exclude_modules: Optional[list[str]] = None,
    metadata_fn: Optional[callable] = None,
) -> nn.Module:
    """Apply barriers to specific modules in a model.
    
    Args:
        model: The model to modify.
        target_modules: List of module names/types to wrap. If None, wraps common types.
        exclude_modules: List of module names/types to exclude from wrapping.
        metadata_fn: Function that takes (module_name, module) and returns metadata dict.
        
    Returns:
        The modified model with barriers applied.
    """
    if target_modules is None:
        # Default target modules (common transformer components)
        target_modules = [
            "Attention", "MultiHeadAttention", "SelfAttention",
            "MLP", "FeedForward", "FFN",
            "LayerNorm", "RMSNorm", "GroupNorm",
            "Linear", "Conv1d", "Conv2d",
        ]

    if exclude_modules is None:
        exclude_modules = []

    def should_wrap_module(name: str, module: nn.Module) -> bool:
        """Determine if a module should be wrapped with barriers."""
        module_type = module.__class__.__name__
        
        # Check exclusions first
        if any(exclude in name or exclude == module_type for exclude in exclude_modules):
            return False
            
        # Check if it's a target module
        return any(target in name or target == module_type for target in target_modules)

    # Apply barriers to matching modules
    for name, module in model.named_modules():
        if should_wrap_module(name, module):
            # Generate metadata if function provided
            metadata = metadata_fn(name, module) if metadata_fn else {"module_name": name}
            
            # Replace the module with a barrier-wrapped version
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            
            if parent_name:
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
                
            wrapped_module = wrap_module_with_barrier(
                module=module,
                region_name=f"{module.__class__.__name__.lower()}_{module_name}",
                metadata=metadata,
            )
            
            setattr(parent_module, module_name, wrapped_module)

    return model


class AutoBarrierModel(nn.Module):
    """A model wrapper that automatically applies barriers to specified module types."""

    def __init__(
        self,
        model: nn.Module,
        auto_barrier_types: Optional[list[str]] = None,
        barrier_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an auto-barrier model.
        
        Args:
            model: The base model to wrap.
            auto_barrier_types: Module types to automatically wrap with barriers.
            barrier_metadata: Default metadata for all barriers.
        """
        super().__init__()
        self.base_model = model
        self.barrier_metadata = barrier_metadata or {}
        
        if auto_barrier_types is None:
            auto_barrier_types = ["Attention", "MLP", "LayerNorm", "RMSNorm"]
        
        # Apply barriers to the model
        apply_barriers_to_model(
            self.base_model,
            target_modules=auto_barrier_types,
            metadata_fn=lambda name, module: {
                **self.barrier_metadata,
                "module_name": name,
                "module_type": module.__class__.__name__,
            }
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the barrier-enhanced model."""
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to the base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_barrier_config(
    module_types: Optional[list[str]] = None,
    custom_regions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a configuration for barrier application.
    
    Args:
        module_types: List of module types to automatically wrap.
        custom_regions: Custom barrier configurations for specific modules.
        
    Returns:
        Barrier configuration dictionary.
    """
    config = {
        "auto_barrier_types": module_types or [
            "Attention", "MLP", "LayerNorm", "RMSNorm", "Linear"
        ],
        "custom_regions": custom_regions or {},
        "global_metadata": {
            "auto_generated": True,
            "barrier_version": "1.0",
        }
    }
    return config


# Example usage functions
def wrap_attention_modules(model: nn.Module) -> nn.Module:
    """Convenience function to wrap attention modules with barriers."""
    return apply_barriers_to_model(
        model,
        target_modules=["Attention", "MultiHeadAttention", "SelfAttention"],
        metadata_fn=lambda name, module: {
            "region": "attention",
            "module_name": name,
            "num_heads": getattr(module, "num_attention_heads", None),
        }
    )


def wrap_mlp_modules(model: nn.Module) -> nn.Module:
    """Convenience function to wrap MLP modules with barriers."""
    return apply_barriers_to_model(
        model,
        target_modules=["MLP", "FeedForward", "FFN"],
        metadata_fn=lambda name, module: {
            "region": "mlp",
            "module_name": name,
            "hidden_size": getattr(module, "hidden_size", None),
        }
    )


def wrap_normalization_modules(model: nn.Module) -> nn.Module:
    """Convenience function to wrap normalization modules with barriers."""
    return apply_barriers_to_model(
        model,
        target_modules=["LayerNorm", "RMSNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d"],
        metadata_fn=lambda name, module: {
            "region": "normalization",
            "module_name": name,
            "normalized_shape": getattr(module, "normalized_shape", None),
        }
    )
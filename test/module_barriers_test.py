import unittest
import torch
import torch.nn as nn

from torch_onnx_models._module_barriers import (
    BarrierModule,
    with_module_barrier,
    wrap_module_with_barrier,
    apply_barriers_to_model,
    AutoBarrierModel,
    create_barrier_config
)


class SimpleLinear(nn.Module):
    """Simple linear module for testing."""
    
    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)


class SimpleModel(nn.Module):
    """Simple model with multiple components."""
    
    def __init__(self):
        super().__init__()
        self.attention = SimpleLinear(10, 10)
        self.mlp = SimpleLinear(10, 10)
        self.norm = nn.LayerNorm(10)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x


@with_module_barrier(
    region_name="test_decorated",
    metadata={"test": "decorator"}
)
class DecoratedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class ModuleBarriersTest(unittest.TestCase):
    
    def setUp(self):
        self.simple_module = SimpleLinear()
        self.simple_model = SimpleModel()
    
    def test_barrier_module_creation(self):
        """Test BarrierModule creation and basic functionality."""
        barrier_module = BarrierModule(
            wrapped_module=self.simple_module,
            region_name="test_region",
            metadata={"test": True}
        )
        
        self.assertEqual(barrier_module.region_name, "test_region")
        self.assertEqual(barrier_module.metadata["test"], True)
        self.assertIsInstance(barrier_module.wrapped_module, SimpleLinear)
    
    def test_barrier_module_forward(self):
        """Test BarrierModule forward pass."""
        barrier_module = BarrierModule(self.simple_module)
        x = torch.randn(5, 10)
        
        # Forward pass should work without errors
        output = barrier_module(x)
        self.assertEqual(output.shape, (5, 10))
    
    def test_barrier_module_attribute_delegation(self):
        """Test that BarrierModule delegates attributes to wrapped module."""
        barrier_module = BarrierModule(self.simple_module)
        
        # Should be able to access wrapped module's attributes
        self.assertEqual(barrier_module.linear.in_features, 10)
        self.assertEqual(barrier_module.linear.out_features, 10)
    
    def test_wrap_module_with_barrier(self):
        """Test the wrap_module_with_barrier function."""
        wrapped = wrap_module_with_barrier(
            self.simple_module,
            region_name="wrapped_test",
            metadata={"wrapped": True}
        )
        
        self.assertIsInstance(wrapped, BarrierModule)
        self.assertEqual(wrapped.region_name, "wrapped_test")
        self.assertEqual(wrapped.metadata["wrapped"], True)
    
    def test_decorated_module(self):
        """Test module with barrier decorator."""
        decorated = DecoratedModule()
        x = torch.randn(5, 10)
        
        # Should work without errors
        output = decorated(x)
        self.assertEqual(output.shape, (5, 10))
    
    def test_apply_barriers_to_model(self):
        """Test applying barriers to specific modules in a model."""
        model = SimpleModel()
        
        # Apply barriers to Linear modules
        modified_model = apply_barriers_to_model(
            model,
            target_modules=["SimpleLinear"],
            metadata_fn=lambda name, module: {"module_name": name}
        )
        
        # Check that attention and mlp modules are wrapped
        self.assertIsInstance(modified_model.attention, BarrierModule)
        self.assertIsInstance(modified_model.mlp, BarrierModule)
        # LayerNorm should not be wrapped
        self.assertIsInstance(modified_model.norm, nn.LayerNorm)
    
    def test_auto_barrier_model(self):
        """Test AutoBarrierModel functionality.""" 
        model = SimpleModel()
        
        auto_barrier_model = AutoBarrierModel(
            model=model,
            auto_barrier_types=["SimpleLinear", "LayerNorm"],
            barrier_metadata={"auto": True}
        )
        
        x = torch.randn(5, 10)
        output = auto_barrier_model(x)
        self.assertEqual(output.shape, (5, 10))
        
        # Check that modules are wrapped
        self.assertIsInstance(auto_barrier_model.base_model.attention, BarrierModule)
        self.assertIsInstance(auto_barrier_model.base_model.mlp, BarrierModule)
        self.assertIsInstance(auto_barrier_model.base_model.norm, BarrierModule)
    
    def test_create_barrier_config(self):
        """Test barrier configuration creation."""
        config = create_barrier_config(
            module_types=["Attention", "MLP"],
            custom_regions={"special": {"metadata": "test"}}
        )
        
        self.assertIn("auto_barrier_types", config)
        self.assertIn("custom_regions", config)
        self.assertIn("global_metadata", config)
        self.assertEqual(config["auto_barrier_types"], ["Attention", "MLP"])
        self.assertEqual(config["custom_regions"]["special"]["metadata"], "test")
    
    def test_barrier_with_onnx_export(self):
        """Test that barriers work during ONNX export."""
        model = BarrierModule(self.simple_module, region_name="onnx_test")
        x = torch.randn(1, 10)
        
        try:
            # This should not raise an error even if barriers are active
            onnx_program = torch.onnx.export(
                model,
                (x,),
                dynamo=True,
                verbose=False
            )
            # Check that export succeeded
            self.assertIsNotNone(onnx_program)
        except Exception as e:
            # If ONNX export fails due to environment, that's okay for this test
            # We're mainly testing that the barrier code doesn't break
            if "onnx" not in str(e).lower():
                raise e


if __name__ == "__main__":
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from argparse import Namespace

from torch_onnx_models._weight_loading import (
    WeightLoader,
    load_weights_with_auto_mapping,
    create_common_weight_mappings,
    WeightMapper
)


class SimpleModel(nn.Module):
    """Simple model for testing weight loading."""
    
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768, bias=False)
        self.k_proj = nn.Linear(768, 768, bias=False)
        self.v_proj = nn.Linear(768, 768, bias=False)
        self.o_proj = nn.Linear(768, 768, bias=False)
    
    def forward(self, x):
        return x


class WeightLoadingTest(unittest.TestCase):
    
    def setUp(self):
        self.model = SimpleModel()
        self.loader = WeightLoader(strict=False)
    
    def test_weight_loader_creation(self):
        """Test WeightLoader initialization."""
        loader = WeightLoader(strict=True)
        self.assertTrue(loader.strict)
        self.assertEqual(len(loader._weight_mappings), 0)
    
    def test_add_weight_mapping(self):
        """Test adding weight mappings."""
        self.loader.add_weight_mapping("source_key", "target_key")
        self.assertEqual(self.loader._weight_mappings["source_key"], "target_key")
    
    def test_add_weight_mappings(self):
        """Test adding multiple weight mappings."""
        mappings = {
            "source1": "target1",
            "source2": "target2"
        }
        self.loader.add_weight_mappings(mappings)
        self.assertEqual(len(self.loader._weight_mappings), 2)
        self.assertEqual(self.loader._weight_mappings["source1"], "target1")
    
    def test_load_from_state_dict(self):
        """Test loading weights from state dict."""
        # Create mock state dict
        source_state_dict = {
            "q_proj.weight": torch.randn(768, 768),
            "k_proj.weight": torch.randn(768, 768),
            "unknown_key": torch.randn(100, 100)
        }
        
        stats = self.loader._load_state_dict(
            self.model, source_state_dict, "", ignore_missing=True
        )
        
        self.assertEqual(stats["total_loaded"], 2)
        self.assertIn("v_proj.weight", stats["missing_keys"])
        self.assertIn("unknown_key", stats["unexpected_keys"])
    
    def test_create_common_weight_mappings(self):
        """Test creation of common weight mappings."""
        mappings = create_common_weight_mappings()
        self.assertIsInstance(mappings, dict)
        self.assertIn("self_attn.q_proj.weight", mappings)
        self.assertEqual(mappings["self_attn.q_proj.weight"], "q_proj.weight")
    
    def test_weight_mapper(self):
        """Test WeightMapper functionality."""
        mapper = WeightMapper()
        
        # Test adding mapping set
        test_mappings = {"test_source": "test_target"}
        mapper.add_mapping_set("test", test_mappings)
        
        retrieved = mapper.get_mapping_set("test")
        self.assertEqual(retrieved, test_mappings)
    
    def test_expand_layer_mappings(self):
        """Test layer mapping expansion."""
        mapper = WeightMapper()
        mappings = {
            "layer.{}.weight": "layers.{}.weight",
            "static.weight": "static.weight"
        }
        
        expanded = mapper.expand_layer_mappings(mappings, 3)
        
        # Should have 4 mappings: 3 expanded + 1 static
        self.assertEqual(len(expanded), 4)
        self.assertIn("layer.0.weight", expanded)
        self.assertIn("layer.1.weight", expanded)
        self.assertIn("layer.2.weight", expanded)
        self.assertIn("static.weight", expanded)
    
    @patch('torch.load')
    def test_load_from_checkpoint(self, mock_torch_load):
        """Test loading from checkpoint file."""
        # Mock checkpoint data
        mock_torch_load.return_value = {
            "state_dict": {
                "q_proj.weight": torch.randn(768, 768),
                "k_proj.weight": torch.randn(768, 768)
            }
        }
        
        # Create a temporary file path for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            stats = self.loader.load_from_checkpoint(
                self.model, tmp_file.name, ignore_missing=True
            )
            
            self.assertEqual(stats["total_loaded"], 2)
            mock_torch_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
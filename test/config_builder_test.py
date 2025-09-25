import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from torch_onnx_models._config_builder import (
    ModelConfig,
    AttentionConfig,
    MLPConfig,
    TransformerConfig,
    ConfigBuilder,
    ModelFactory,
    quick_config_from_hf
)


class ConfigBuilderTest(unittest.TestCase):
    
    def setUp(self):
        self.builder = ConfigBuilder()
        self.factory = ModelFactory()
    
    def test_model_config_basic(self):
        """Test basic ModelConfig functionality."""
        config = ModelConfig(test_param=123, another_param="test")
        
        self.assertEqual(config.test_param, 123)
        self.assertEqual(config.another_param, "test")
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["test_param"], 123)
        self.assertEqual(config_dict["another_param"], "test")
        
        # Test to_namespace
        namespace = config.to_namespace()
        self.assertIsInstance(namespace, Namespace)
        self.assertEqual(namespace.test_param, 123)
    
    def test_model_config_from_dict(self):
        """Test creating ModelConfig from dictionary."""
        config_dict = {"param1": 1, "param2": "value"}
        config = ModelConfig.from_dict(config_dict)
        
        self.assertEqual(config.param1, 1)
        self.assertEqual(config.param2, "value")
    
    def test_model_config_json_serialization(self):
        """Test JSON save/load for ModelConfig."""
        config = ModelConfig(test_param=123, string_param="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            
            # Load it back
            loaded_config = ModelConfig.from_json(f.name)
            self.assertEqual(loaded_config.test_param, 123)
            self.assertEqual(loaded_config.string_param, "test")
    
    def test_attention_config(self):
        """Test AttentionConfig creation and defaults."""
        config = AttentionConfig(
            hidden_size=1024,
            num_attention_heads=16,
            attention_bias=False
        )
        
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_key_value_heads, 16)  # Should default to num_attention_heads
        self.assertEqual(config.head_dim, 64)  # 1024 // 16
        self.assertFalse(config.attention_bias)
    
    def test_mlp_config(self):
        """Test MLPConfig creation and defaults."""
        config = MLPConfig(
            hidden_size=768,
            hidden_act="gelu"
        )
        
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 3072)  # 4 * 768
        self.assertEqual(config.hidden_act, "gelu")
        self.assertTrue(config.bias)  # Default should be True
    
    def test_transformer_config(self):
        """Test TransformerConfig creation and component extraction."""
        config = TransformerConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072
        )
        
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.intermediate_size, 3072)
        
        # Test component config extraction
        attention_config = config.get_attention_config()
        self.assertIsInstance(attention_config, AttentionConfig)
        self.assertEqual(attention_config.hidden_size, 768)
        self.assertEqual(attention_config.num_attention_heads, 12)
        
        mlp_config = config.get_mlp_config()
        self.assertIsInstance(mlp_config, MLPConfig)
        self.assertEqual(mlp_config.hidden_size, 768)
        self.assertEqual(mlp_config.intermediate_size, 3072)
    
    def test_config_builder_llama(self):
        """Test ConfigBuilder with LLaMA-style configuration."""
        hf_config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "num_key_value_heads": 32
        }
        
        config = self.builder.from_huggingface_config(hf_config)
        
        self.assertIsInstance(config, TransformerConfig)
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(config.attention_bias)  # LLaMA doesn't use bias
        self.assertFalse(config.mlp_bias)
    
    def test_config_builder_mistral(self):
        """Test ConfigBuilder with Mistral-style configuration."""
        hf_config = {
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768
        }
        
        config = self.builder.from_huggingface_config(hf_config)
        
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.max_position_embeddings, 32768)
    
    def test_config_builder_phi(self):
        """Test ConfigBuilder with Phi-style configuration.""" 
        hf_config = {
            "model_type": "phi",
            "vocab_size": 51200,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_hidden_layers": 24,
            "intermediate_size": 8192,
            "attention_bias": True,
            "mlp_bias": True
        }
        
        config = self.builder.from_huggingface_config(hf_config)
        
        self.assertEqual(config.vocab_size, 51200)
        self.assertTrue(config.attention_bias)  # Phi uses bias
        self.assertTrue(config.mlp_bias)
    
    def test_config_builder_generic(self):
        """Test ConfigBuilder with unknown model type (generic fallback)."""
        hf_config = {
            "model_type": "unknown",
            "vocab_size": 50000,
            "hidden_size": 512,
            "num_attention_heads": 8
        }
        
        config = self.builder.from_huggingface_config(hf_config)
        
        # Should use generic defaults
        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.num_hidden_layers, 12)  # Default
    
    def test_config_builder_from_json_file(self):
        """Test loading config from JSON file."""
        hf_config = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_attention_heads": 16
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(hf_config, f)
            f.flush()
            
            config = self.builder.from_huggingface_config(f.name)
            self.assertEqual(config.hidden_size, 2048)
            self.assertEqual(config.num_attention_heads, 16)
    
    def test_custom_model_type_registration(self):
        """Test registering custom model type builder."""
        def custom_builder(hf_config):
            return TransformerConfig(
                hidden_size=hf_config["custom_hidden_size"],
                num_attention_heads=hf_config["custom_heads"]
            )
        
        self.builder.register_model_type("custom", custom_builder)
        
        hf_config = {
            "model_type": "custom",
            "custom_hidden_size": 999,
            "custom_heads": 13
        }
        
        config = self.builder.from_huggingface_config(hf_config)
        self.assertEqual(config.hidden_size, 999)
        self.assertEqual(config.num_attention_heads, 13)
    
    def test_model_factory_attention_creation(self):
        """Test ModelFactory attention creation."""
        config = AttentionConfig(hidden_size=768, num_attention_heads=12)
        
        # This would normally create an Attention module
        # For testing, we'll just verify the method exists and can be called
        try:
            attention = self.factory.create_attention(config)
            # If components are available, this should work
        except ImportError:
            # If components not available, that's expected in isolated testing
            pass
    
    def test_model_factory_mlp_creation(self):
        """Test ModelFactory MLP creation."""
        config = MLPConfig(hidden_size=768, intermediate_size=3072)
        
        try:
            mlp = self.factory.create_mlp(config)
            # If components are available, this should work
        except ImportError:
            # If components not available, that's expected in isolated testing
            pass
    
    def test_model_factory_with_transformer_config(self):
        """Test ModelFactory with TransformerConfig."""
        config = TransformerConfig(hidden_size=768, num_attention_heads=12)
        
        try:
            attention = self.factory.create_attention(config)
            mlp = self.factory.create_mlp(config)
        except ImportError:
            # Expected if components not available
            pass


if __name__ == "__main__":
    unittest.main()
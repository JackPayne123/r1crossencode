#!/usr/bin/env python
"""
Example demonstrating direct compatibility with the sae_vis API
This example follows the usage pattern from the original demo
"""

import os
import sys
import torch
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our compatibility layer
from sae_vis_adapter.sae_vis_compat import (
    create_sae_vis_compatible_encoder,
    setup_sae_vis_data,
    import_sae_vis_modules
)

def example_direct_sae_vis_usage():
    """
    Example of direct sae_vis usage, following your demo pattern
    """
    # Path to the sae_vis library
    SAE_VIS_PATH = "sae_vis-crosscoder-vis/sae_vis"
    
    # Import the ObservableModel from your implementation
    try:
        # Try to import from crosscoder-model-diff-replication
        sys.path.append("crosscoder-model-diff-replication")
        from utils import ObservableModel
    except ImportError:
        # If that fails, use our implementation
        from sae_vis_adapter.observable_model import ObservableModel
    
    try:
        # Import your CrossCoder implementation
        from crosscoder import CrossCoder as HFCrossCoder
    except ImportError:
        # Sample placeholder
        print("Could not import CrossCoder. Using a placeholder for demo purposes.")
        from sae_vis_adapter.adapter import CrossCoderAdapter as HFCrossCoder
    
    # Import sae_vis modules
    sae_vis_modules = import_sae_vis_modules(SAE_VIS_PATH)
    
    # Step 1: Load or create models (simplified for example)
    print("Loading models...")
    
    # For this example, we'll create dummy models
    # In practice, you would load your actual models
    try:
        # Use transformers to load models
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load the base model (e.g., GPT-2)
        base_model = ObservableModel(
            AutoModelForCausalLM.from_pretrained("gpt2"),
            AutoTokenizer.from_pretrained("gpt2")
        )
        
        # For demonstration, use the same model for cot_model
        cot_model = base_model
        
    except (ImportError, Exception) as e:
        print(f"Error loading models: {e}")
        print("Creating placeholder models for demonstration")
        
        # Create dummy base_model and cot_model with a cfg attribute
        class DummyModel:
            def __init__(self):
                self.cfg = DummyConfig()
                self._model = self
                self.name_or_path = "dummy_model"
                
            def parameters(self):
                return iter([torch.randn(1)])
                
            def named_modules(self):
                return [("", self)]
                
            def __call__(self, *args, **kwargs):
                return None
                
        class DummyConfig:
            def __init__(self):
                self.hidden_size = 768
                self.num_hidden_layers = 12
                
        class DummyTokenizer:
            def __init__(self):
                pass
                
            def encode(self, text, return_tensors="pt"):
                # Return a tensor of token IDs
                return torch.randint(0, 1000, (1, len(text.split())))
                
            def decode(self, token_ids):
                # Return a placeholder text
                return "Decoded text"
        
        # Create dummy ObservableModel
        if "ObservableModel" not in locals():
            from sae_vis_adapter.observable_model import ObservableModel
            
        base_model = ObservableModel(DummyModel(), DummyTokenizer())
        cot_model = ObservableModel(DummyModel(), DummyTokenizer())
    
    # Step 2: Create or load CrossCoder
    print("Creating CrossCoder...")
    try:
        # In practice, you would load your trained CrossCoder
        # For this example, we'll create a placeholder
        
        # Configure CrossCoder
        d_in = getattr(base_model, "cfg", None)
        if d_in is not None:
            d_in = d_in.hidden_size
        else:
            d_in = 768  # Default for demonstration
            
        dict_size = 16384  # 2^14, typical size
        
        # Create CrossCoder
        crosscoder_cfg = {
            "d_in": d_in,
            "dict_size": dict_size,
            "hook_point": "model.layers.8.attention.output",  # Example hook point
            "device": "cpu"
        }
        
        hf_crosscoder = HFCrossCoder(crosscoder_cfg)
        
        # Initialize with random weights for demonstration
        hf_crosscoder.W_enc = torch.nn.Parameter(torch.randn(2, d_in, dict_size))
        hf_crosscoder.W_dec = torch.nn.Parameter(torch.randn(dict_size, 2, d_in))
        hf_crosscoder.b_enc = torch.nn.Parameter(torch.zeros(dict_size))
        hf_crosscoder.b_dec = torch.nn.Parameter(torch.zeros(2, d_in))
        
    except Exception as e:
        print(f"Error creating CrossCoder: {e}")
        print("Creating a placeholder CrossCoder for demonstration")
        
        # Create a minimal CrossCoder for demonstration
        class PlaceholderCrossCoder:
            def __init__(self):
                self.cfg = {
                    "d_in": 768,
                    "dict_size": 16384,
                    "hook_point": "model.layers.8.attention.output",
                    "l1_coeff": 3e-4
                }
                self.W_enc = torch.nn.Parameter(torch.randn(2, 768, 16384))
                self.W_dec = torch.nn.Parameter(torch.randn(16384, 2, 768))
                self.b_enc = torch.nn.Parameter(torch.zeros(16384))
                self.b_dec = torch.nn.Parameter(torch.zeros(2, 768))
                
            def state_dict(self):
                return {
                    "W_enc": self.W_enc,
                    "W_dec": self.W_dec,
                    "b_enc": self.b_enc,
                    "b_dec": self.b_dec
                }
                
            def to(self, *args, **kwargs):
                return self
        
        hf_crosscoder = PlaceholderCrossCoder()
    
    # Step 3: Create sae_vis compatible CrossCoder
    print("Creating sae_vis compatible CrossCoder...")
    CrossCoderConfig = sae_vis_modules["model_fns"].CrossCoderConfig
    
    # Following your demo pattern:
    encoder_cfg = CrossCoderConfig(
        d_in=base_model.cfg.hidden_size, 
        d_hidden=hf_crosscoder.cfg["dict_size"], 
        apply_b_dec_to_input=False
    )
    
    # Create the adapter
    sae_vis_encoder = create_sae_vis_compatible_encoder(
        hf_crosscoder=hf_crosscoder,
        d_in=base_model.cfg.hidden_size,
        d_hidden=hf_crosscoder.cfg["dict_size"],
        device="cpu",
        dtype=torch.float32
    )
    
    # Load weights (would typically come from your trained model)
    sae_vis_encoder.load_state_dict(hf_crosscoder.state_dict())
    
    # Step 4: Create SaeVisConfig
    test_feature_idx = [0, 1, 2]  # Example feature indices
    
    # Step 5: Generate some dummy tokenized data for demonstration
    print("Generating tokenized data...")
    num_sequences = 10
    seq_length = 20
    dummy_tokens = {
        "tokens_A": torch.randint(0, 1000, (num_sequences, seq_length)),
        "tokens_B": torch.randint(0, 1000, (num_sequences, seq_length))
    }
    
    # Step 6: Create SaeVisData
    print("Creating SaeVisData...")
    hook_point = hf_crosscoder.cfg["hook_point"]
    
    # Setup SaeVisData using our compatibility layer
    sae_vis_data = setup_sae_vis_data(
        hf_crosscoder=hf_crosscoder,
        model_A=base_model,
        model_B=cot_model,
        tokens=dummy_tokens,
        hook_point=hook_point,
        features=test_feature_idx,
        sae_vis_path=SAE_VIS_PATH,
        device="cpu",
        dtype=torch.float32
    )
    
    # Step 7: Use SaeVisData to create visualizations
    # In practice, you would call functions from html_fns
    print("SaeVisData created successfully!")
    print(f"Features: {test_feature_idx}")
    print(f"Hook point: {hook_point}")
    print("Ready to generate visualizations with sae_vis!")
    
    return sae_vis_data

if __name__ == "__main__":
    try:
        sae_vis_data = example_direct_sae_vis_usage()
    except Exception as e:
        import traceback
        print(f"Error in example: {e}")
        traceback.print_exc() 
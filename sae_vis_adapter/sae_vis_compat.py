import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import importlib
import os
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple

from .adapter import CrossCoderAdapter, CrossCoderConfig
from .transformer_lens_adapter import TransformerLensWrapperAdapter
from .observable_model import ObservableModel

# Function to dynamically import sae_vis modules
def import_sae_vis_modules(sae_vis_path: str) -> Dict[str, Any]:
    """
    Dynamically import sae_vis modules from the specified path
    
    Args:
        sae_vis_path: Path to the sae_vis package root directory
        
    Returns:
        Dictionary mapping module names to imported modules
    """
    try:
        # Add the parent directory to sys.path so Python can find the sae_vis package
        if sae_vis_path not in sys.path:
            sys.path.append(sae_vis_path)
        
        # For debugging
        print(f"Python path now includes: {sae_vis_path}")
        print(f"Full sys.path: {sys.path}")
        
        # Try to verify the directory structure
        sae_vis_dir = os.path.join(sae_vis_path, "sae_vis")
        if os.path.exists(sae_vis_dir):
            print(f"Found sae_vis directory at {sae_vis_dir}")
            print(f"Contents: {os.listdir(sae_vis_dir)}")
        else:
            print(f"Warning: Could not find sae_vis directory at {sae_vis_dir}")
            # If sae_vis directory doesn't exist at this path, maybe the path itself is the sae_vis directory
            if os.path.exists(os.path.join(sae_vis_path, "model_fns.py")):
                print(f"Found model_fns.py directly in {sae_vis_path}")
                # In this case, use the path directly without 'sae_vis' prefix
                # Add the directory containing the Python files to sys.path
                sys.path.append(os.path.dirname(sae_vis_path))
                print(f"Using directory {sae_vis_path} as a module")
                
                # Try importing modules directly by name
                model_fns = importlib.import_module("model_fns")
                data_config_classes = importlib.import_module("data_config_classes")
                data_storing_fns = importlib.import_module("data_storing_fns")
                data_fetching_fns = importlib.import_module("data_fetching_fns")
                
                # Return the modules
                return {
                    "model_fns": model_fns,
                    "data_config_classes": data_config_classes,
                    "data_storing_fns": data_storing_fns,
                    "data_fetching_fns": data_fetching_fns
                }
        
        # Import the original modules for configuration types
        sae_vis_model_fns = importlib.import_module("sae_vis.model_fns")
        sae_vis_data_config = importlib.import_module("sae_vis.data_config_classes")
        sae_vis_data_storing = importlib.import_module("sae_vis.data_storing_fns")
        sae_vis_data_fetching = importlib.import_module("sae_vis.data_fetching_fns")
        
        return {
            "model_fns": sae_vis_model_fns,
            "data_config_classes": sae_vis_data_config,
            "data_storing_fns": sae_vis_data_storing,
            "data_fetching_fns": sae_vis_data_fetching
        }
    except Exception as e:
        print(f"Error importing sae_vis modules: {e}")
        print(f"Make sure the path '{sae_vis_path}' is correct and contains the sae_vis package.")
        
        # List the actual directory contents to help diagnose
        try:
            print(f"Contents of {sae_vis_path}: {os.listdir(sae_vis_path)}")
        except:
            print(f"Could not list contents of {sae_vis_path}")
            
        raise


class DirectCrossCoderAdapter(nn.Module):
    """
    Direct adapter for HuggingFace CrossCoder to sae_vis.model_fns.CrossCoder
    Works as a drop-in replacement for sae_vis.model_fns.CrossCoder
    """
    def __init__(self, cfg, hf_crosscoder=None):
        """
        Initialize with either a configuration or a HuggingFace CrossCoder
        
        Args:
            cfg: Either CrossCoderConfig from sae_vis or our own adapter config
            hf_crosscoder: Optional HuggingFace CrossCoder instance
        """
        super().__init__()
        self.cfg = cfg
        
        # Initialize attributes
        self.W_enc = None
        self.W_dec = None
        self.b_enc = None
        self.b_dec = None
        
        # If a HuggingFace CrossCoder is provided, use its weights
        if hf_crosscoder is not None:
            self.init_from_hf_crosscoder(hf_crosscoder)
    
    def init_from_hf_crosscoder(self, hf_crosscoder):
        """
        Initialize weights from a HuggingFace CrossCoder
        
        Args:
            hf_crosscoder: HuggingFace CrossCoder instance
        """
        # Get weights from the HuggingFace CrossCoder
        self.W_enc = nn.Parameter(hf_crosscoder.W_enc.detach().clone())
        self.W_dec = nn.Parameter(hf_crosscoder.W_dec.detach().clone())
        self.b_enc = nn.Parameter(hf_crosscoder.b_enc.detach().clone())
        self.b_dec = nn.Parameter(hf_crosscoder.b_dec.detach().clone())
        
        # Set device and dtype to match
        self.to(hf_crosscoder.W_enc.device)
        
        # Get the L1 coefficient from the HuggingFace CrossCoder if available
        if hasattr(hf_crosscoder, "cfg") and isinstance(hf_crosscoder.cfg, dict):
            self.l1_coeff = hf_crosscoder.cfg.get("l1_coeff", 3e-4)
        else:
            self.l1_coeff = getattr(self.cfg, "l1_coeff", 3e-4)
        
        return self
    
    def encode(self, x, apply_relu=True):
        """
        Encode input to activations, matching sae_vis.model_fns.CrossCoder.encode
        
        Args:
            x: Input tensor [batch, n_models, d_model]
            apply_relu: Whether to apply ReLU
            
        Returns:
            Encoded activations
        """
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            return F.relu(x_enc + self.b_enc)
        else:
            return x_enc + self.b_enc
    
    def decode(self, acts):
        """
        Decode activations to reconstructed input, matching sae_vis.model_fns.CrossCoder.decode
        
        Args:
            acts: Activations tensor [batch, d_hidden]
            
        Returns:
            Reconstructed input tensor [batch, n_models, d_model]
        """
        x_reconstruct = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        
        if hasattr(self.cfg, "apply_b_dec_to_input") and self.cfg.apply_b_dec_to_input:
            return x_reconstruct + self.b_dec
        else:
            return x_reconstruct
    
    def forward(self, x):
        """
        Forward pass, matching sae_vis.model_fns.CrossCoder.forward
        
        Args:
            x: Input tensor [batch, n_models, d_model]
            
        Returns:
            Reconstructed input
        """
        acts = self.encode(x)
        return self.decode(acts)
    
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        """Implementation of the optimization step for orthogonal decoder directions"""
        if self.W_dec.grad is not None:
            W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj


class ModelWrapper:
    """
    Wrapper to make a model compatible with sae_vis
    
    This is a simpler version of TransformerLensWrapperAdapter that adheres
    more directly to the expected interface in sae_vis
    """
    def __init__(self, observable_model, hook_point):
        """
        Initialize the wrapper
        
        Args:
            observable_model: ObservableModel instance
            hook_point: Hook point to extract activations from
        """
        self.model = observable_model
        self.hook_point = hook_point
        self.tokenizer = observable_model.tokenizer
        self.device = next(observable_model._model.parameters()).device
    
    def get_hidden_states(self, tokens, layer):
        """
        Get hidden states from a specific layer, matching sae_vis expected interface
        
        Args:
            tokens: Input tokens
            layer: Layer to get hidden states from (ignored, we use hook_point)
            
        Returns:
            Hidden states tensor
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=self.hook_point)
            return cache[self.hook_point]
    
    def to(self, device):
        """Move model to device"""
        self.model._model.to(device)
        self.device = device
        return self


def create_sae_vis_compatible_encoder(hf_crosscoder, d_in, d_hidden=None, 
                                      apply_b_dec_to_input=False, l1_coeff=3e-4,
                                      device="cuda:0", dtype=torch.float32):
    """
    Create a sae_vis compatible encoder from a HuggingFace CrossCoder
    
    Args:
        hf_crosscoder: HuggingFace CrossCoder instance
        d_in: Input dimension
        d_hidden: Hidden dimension (if None, inferred from hf_crosscoder)
        apply_b_dec_to_input: Whether to apply b_dec to input
        l1_coeff: L1 regularization coefficient
        device: Device to put the model on
        dtype: Data type to use
        
    Returns:
        DirectCrossCoderAdapter instance compatible with sae_vis
    """
    # Import original CrossCoderConfig
    sae_vis_modules = import_sae_vis_modules()
    orig_CrossCoderConfig = sae_vis_modules["model_fns"].CrossCoderConfig
    
    # Infer d_hidden if not provided
    if d_hidden is None and hf_crosscoder is not None:
        d_hidden = hf_crosscoder.W_enc.shape[2]
    
    # Create config with the same structure as original
    encoder_cfg = orig_CrossCoderConfig(
        d_in=d_in,
        d_hidden=d_hidden,
        apply_b_dec_to_input=apply_b_dec_to_input
    )
    
    # Create the adapter
    sae_vis_encoder = DirectCrossCoderAdapter(encoder_cfg)
    
    # Initialize from HuggingFace CrossCoder if provided
    if hf_crosscoder is not None:
        sae_vis_encoder.init_from_hf_crosscoder(hf_crosscoder)
    
    # Move to specified device and dtype
    sae_vis_encoder = sae_vis_encoder.to(device)
    sae_vis_encoder = sae_vis_encoder.to(dtype)
    
    return sae_vis_encoder


def create_model_wrapper(model, hook_point):
    """
    Create a model wrapper compatible with sae_vis
    
    Args:
        model: Model to wrap (can be ObservableModel or regular HuggingFace model)
        hook_point: Hook point to extract activations from
        
    Returns:
        ModelWrapper instance
    """
    # If already an ObservableModel, use it directly
    if hasattr(model, "run_with_cache") and callable(model.run_with_cache):
        observable_model = model
    else:
        # Create ObservableModel from HuggingFace model
        from .observable_model import ObservableModel
        observable_model = ObservableModel(model, model.tokenizer)
    
    return ModelWrapper(observable_model, hook_point)


def setup_sae_vis_data(hf_crosscoder, model_A, model_B, tokens, 
                       hook_point=None, features=None, sae_vis_path=None,
                       device="cuda:0", dtype=torch.float32):
    """
    Setup SaeVisData with HuggingFace models, matching the demo usage pattern
    
    Args:
        hf_crosscoder: HuggingFace CrossCoder instance
        model_A: First model (ObservableModel or HuggingFace model)
        model_B: Second model (ObservableModel or HuggingFace model) or None
        tokens: Tokenized data
        hook_point: Hook point to use (if None, use from hf_crosscoder.cfg)
        features: List of feature indices to visualize
        sae_vis_path: Path to sae_vis directory
        device: Device to use
        dtype: Data type to use
        
    Returns:
        SaeVisData instance
    """
    # Import sae_vis modules
    sae_vis_modules = import_sae_vis_modules(sae_vis_path)
    SaeVisConfig = sae_vis_modules["data_config_classes"].SaeVisConfig
    SaeVisData = sae_vis_modules["data_storing_fns"].SaeVisData
    
    # Get hook point from CrossCoder config if not provided
    if hook_point is None and hasattr(hf_crosscoder, "cfg"):
        hook_point = hf_crosscoder.cfg.get("hook_point")
    
    if hook_point is None:
        raise ValueError("hook_point must be provided either directly or in hf_crosscoder.cfg")
    
    # Create sae_vis compatible encoder
    encoder = create_sae_vis_compatible_encoder(
        hf_crosscoder,
        d_in=model_A.cfg.hidden_size if hasattr(model_A, "cfg") else None, 
        device=device,
        dtype=dtype
    )
    
    # Create model wrappers
    model_A_wrapper = create_model_wrapper(model_A, hook_point)
    model_B_wrapper = create_model_wrapper(model_B, hook_point) if model_B is not None else None
    
    # Create SaeVisConfig
    sae_vis_config = SaeVisConfig(
        hook_point=hook_point,
        features=features,
        verbose=True,
        minibatch_size_tokens=4,
        minibatch_size_features=16,
    )
    
    # Create SaeVisData
    sae_vis_data = SaeVisData.create(
        encoder=encoder,
        encoder_B=None,
        model_A=model_A_wrapper,
        model_B=model_B_wrapper,
        tokens=tokens,
        cfg=sae_vis_config,
    )
    
    return sae_vis_data 
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import os
import json
from functools import partial

def get_model_hook_points(observable_model) -> List[str]:
    """
    Get all available hook points from an ObservableModel
    
    Args:
        observable_model: The ObservableModel instance
        
    Returns:
        List of available hook points
    """
    # Call the ObservableModel's method to get hook points
    return observable_model.get_available_hook_points()

def discover_hook_point_for_layer(observable_model, layer_num: int, 
                                  hook_type: str = "attention_output") -> str:
    """
    Try to find an appropriate hook point for a specific layer and hook type
    
    Args:
        observable_model: The ObservableModel instance
        layer_num: Layer number to target
        hook_type: Type of hook to look for ("attention_output", "mlp_output", etc.)
        
    Returns:
        Best matching hook point
    """
    hook_points = get_model_hook_points(observable_model)
    
    # Common patterns in different model architectures
    patterns = [
        f"model.layers.{layer_num}.{hook_type}",  # LLaMA-style
        f"transformer.h.{layer_num}.{hook_type}",  # GPT2-style
        f"encoder.layer.{layer_num}.{hook_type}",  # BERT-style
        f"decoder.layers.{layer_num}.{hook_type}"  # T5-style decoder
    ]
    
    # Try to find a matching hook point
    for pattern in patterns:
        matches = [hp for hp in hook_points if pattern in hp]
        if matches:
            return matches[0]
    
    # Fallback - look for any hook point with the layer number and hook type
    layer_str = str(layer_num)
    matches = [hp for hp in hook_points if layer_str in hp and hook_type in hp]
    if matches:
        return matches[0]
    
    raise ValueError(f"Could not find a hook point for layer {layer_num} and type {hook_type}")

def load_crosscoder_checkpoint(path: str, crosscoder_model):
    """
    Load CrossCoder checkpoint
    
    Args:
        path: Path to checkpoint file
        crosscoder_model: CrossCoder model instance
        
    Returns:
        Loaded CrossCoder model
    """
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        crosscoder_model.load_state_dict(checkpoint['model_state_dict'])
        return crosscoder_model
    except Exception as e:
        print(f"Error loading CrossCoder checkpoint: {e}")
        raise

def setup_visualization_data(model_wrapper, crosscoder_adapter, prompt: str) -> Dict[str, Any]:
    """
    Setup data for visualization
    
    Args:
        model_wrapper: The model wrapper adapter
        crosscoder_adapter: The CrossCoder adapter
        prompt: Text prompt to use for visualization
        
    Returns:
        Dictionary with data for visualization
    """
    tokens = model_wrapper.tokenizer.encode(prompt, return_tensors="pt").to(model_wrapper.device)
    
    # Get model activations
    activations = model_wrapper.get_activations(tokens)
    
    # Get CrossCoder features
    with torch.no_grad():
        # Format activations for CrossCoder
        acts_formatted = activations.unsqueeze(1)  # Add n_models dimension
        
        # Run CrossCoder forward
        _, reconstructed, feature_acts, _, _ = crosscoder_adapter(acts_formatted)
    
    return {
        "tokens": tokens,
        "text": prompt,
        "token_strings": [model_wrapper.tokenizer.decode(t) for t in tokens[0]],
        "raw_activations": activations,
        "feature_activations": feature_acts,
        "reconstructed_activations": reconstructed.squeeze(1)
    }

def save_feature_data(feature_data: Dict[str, Any], output_dir: str, feature_idx: int):
    """
    Save feature data to JSON file
    
    Args:
        feature_data: Dictionary with feature data
        output_dir: Directory to save the data
        feature_idx: Index of the feature to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to lists
    serializable_data = {}
    for key, value in feature_data.items():
        if isinstance(value, torch.Tensor):
            # Convert to CPU, detach, and convert to list
            serializable_data[key] = value.cpu().detach().numpy().tolist()
        else:
            serializable_data[key] = value
    
    # Save to JSON file
    with open(os.path.join(output_dir, f"feature_{feature_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)
        
    print(f"Saved feature {feature_idx} data to {output_dir}/feature_{feature_idx}.json")
    
def get_feature_importance(crosscoder_adapter, feature_idx: int) -> torch.Tensor:
    """
    Calculate feature importance based on decoder weights
    
    Args:
        crosscoder_adapter: The CrossCoder adapter
        feature_idx: Index of the feature
        
    Returns:
        Tensor with feature importance scores
    """
    # Extract decoder weights for this feature
    W_dec_feature = crosscoder_adapter.W_dec[feature_idx]
    
    # Calculate L2 norm across model dimension
    importance = W_dec_feature.norm(dim=-1)
    
    return importance 
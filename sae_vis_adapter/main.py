#!/usr/bin/env python
"""
Main script for the CrossCoder visualization adapter
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Make sure we can import from parent directory
sys.path.append(str(Path(__file__).parent.parent))

# Import adapter modules
from sae_vis_adapter.adapter import CrossCoderAdapter, CrossCoderConfig
from sae_vis_adapter.transformer_lens_adapter import TransformerLensWrapperAdapter
from sae_vis_adapter.utils import discover_hook_point_for_layer, load_crosscoder_checkpoint
from sae_vis_adapter.visualization import visualize_top_features

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CrossCoder Visualization Adapter")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the HuggingFace model to use (local path or model name)"
    )
    parser.add_argument(
        "--crosscoder_path", 
        type=str, 
        required=True,
        help="Path to the CrossCoder checkpoint file"
    )
    parser.add_argument(
        "--sae_vis_path", 
        type=str, 
        default="sae_vis-crosscoder-vis/sae_vis",
        help="Path to the sae_vis directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--layer", 
        type=int, 
        default=8,
        help="Layer to visualize features for"
    )
    parser.add_argument(
        "--hook_type", 
        type=str, 
        default="attention_output",
        choices=["attention_output", "mlp_output", "residual"],
        help="Type of hook to use"
    )
    parser.add_argument(
        "--num_features", 
        type=int, 
        default=10,
        help="Number of top features to visualize"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    
    return parser.parse_args()

def load_model_and_observable(model_path, device):
    """
    Load the model and create an ObservableModel wrapper
    
    Args:
        model_path: Path to the model or model name
        device: Device to load the model on
        
    Returns:
        ObservableModel instance
    """
    try:
        # First try to import our custom ObservableModel implementation
        try:
            # Try to import from crosscoder-model-diff-replication
            sys.path.append("crosscoder-model-diff-replication")
            from utils import ObservableModel
        except ImportError:
            # If that fails, look in the current directory
            from observable_model import ObservableModel
        
        # Import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create ObservableModel
        print("Creating ObservableModel wrapper...")
        observable_model = ObservableModel(model, tokenizer)
        
        # Move to device
        observable_model._model.to(device)
        
        return observable_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_crosscoder(crosscoder_path, hook_point, device):
    """
    Load the CrossCoder model
    
    Args:
        crosscoder_path: Path to the CrossCoder checkpoint
        hook_point: Hook point for the CrossCoder
        device: Device to load the model on
        
    Returns:
        CrossCoder instance
    """
    try:
        # First try to import our custom CrossCoder implementation
        try:
            # Try to import from crosscoder-model-diff-replication
            sys.path.append("crosscoder-model-diff-replication")
            from crosscoder import CrossCoder as HFCrossCoder
        except ImportError:
            # If that fails, look in the current directory
            from crosscoder_model import CrossCoder as HFCrossCoder
        
        # Load the CrossCoder checkpoint
        print(f"Loading CrossCoder from {crosscoder_path}...")
        checkpoint = torch.load(crosscoder_path, map_location=torch.device(device))
        
        # Extract configuration from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Infer CrossCoder dimensions from state dict
        W_enc = state_dict['W_enc']
        d_model = W_enc.shape[1]
        d_hidden = W_enc.shape[2]
        
        # Create HuggingFace CrossCoder
        hf_crosscoder = HFCrossCoder(d_in=d_model, d_hidden=d_hidden)
        hf_crosscoder.load_state_dict(state_dict)
        hf_crosscoder.to(device)
        
        # Create adapter configuration
        config = CrossCoderConfig(
            d_in=d_model,
            d_hidden=d_hidden,
            l1_coeff=checkpoint.get('l1_coeff', 3e-4)
        )
        
        # Create adapter
        adapter = CrossCoderAdapter(hf_crosscoder, config)
        
        return adapter
    
    except Exception as e:
        print(f"Error loading CrossCoder: {e}")
        raise

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and create ObservableModel
    observable_model = load_model_and_observable(args.model_path, args.device)
    
    # Get hook point for the specified layer
    hook_point = discover_hook_point_for_layer(
        observable_model, 
        args.layer, 
        args.hook_type
    )
    print(f"Using hook point: {hook_point}")
    
    # Create transformer lens wrapper
    model_wrapper = TransformerLensWrapperAdapter(observable_model, hook_point)
    
    # Load CrossCoder
    crosscoder_adapter = load_crosscoder(args.crosscoder_path, hook_point, args.device)
    
    # Sample prompts for visualization
    sample_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "I think that artificial intelligence is going to transform how we live and work in the next decade.",
        "In computer programming, a function is a block of code designed to perform a specific task.",
        "The history of science is marked by paradigm shifts where new theories replace older ones.",
        "Machine learning models learn patterns from data without being explicitly programmed."
    ]
    
    # Create visualizations
    visualization_files = visualize_top_features(
        model_wrapper,
        crosscoder_adapter,
        args.num_features,
        sample_prompts,
        args.output_dir,
        args.sae_vis_path
    )
    
    print(f"Visualizations saved to {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'index.html')} to view the visualizations")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 
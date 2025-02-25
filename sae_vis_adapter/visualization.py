from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import numpy as np
import os
import sys
import json
from pathlib import Path
import importlib.util

# Import from adapter modules
from .utils import setup_visualization_data, save_feature_data, get_feature_importance

# Function to import sae_vis dynamically
def import_sae_vis(sae_vis_path: str):
    """
    Dynamically import the sae_vis library
    
    Args:
        sae_vis_path: Path to the sae_vis directory
    
    Returns:
        Imported sae_vis module
    """
    try:
        # Add the sae_vis directory to the path
        if sae_vis_path not in sys.path:
            sys.path.append(sae_vis_path)
            
        # Import main modules
        import sae_vis
        from sae_vis import html_fns, utils_fns, data_config_classes
        
        return {
            'sae_vis': sae_vis,
            'html_fns': html_fns,
            'utils_fns': utils_fns,
            'data_config_classes': data_config_classes
        }
    except ImportError as e:
        print(f"Error importing sae_vis: {e}")
        print(f"Make sure the path '{sae_vis_path}' is correct and contains the sae_vis package.")
        raise

def prepare_feature_visualization(
    model_wrapper, 
    crosscoder_adapter,
    feature_idx: int,
    sample_prompts: List[str],
    output_dir: str,
    sae_vis_modules: Dict
):
    """
    Prepare visualization data for a specific feature
    
    Args:
        model_wrapper: Model wrapper adapter
        crosscoder_adapter: CrossCoder adapter
        feature_idx: Index of the feature to visualize
        sample_prompts: List of text prompts to use as examples
        output_dir: Directory to save visualization data
        sae_vis_modules: Dictionary of imported sae_vis modules
    
    Returns:
        Path to the generated visualization HTML file
    """
    html_fns = sae_vis_modules['html_fns']
    utils_fns = sae_vis_modules['utils_fns']
    data_config_classes = sae_vis_modules['data_config_classes']
    
    # Create output directory
    feature_dir = os.path.join(output_dir, f"feature_{feature_idx}")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Process each sample prompt
    all_feature_data = []
    for i, prompt in enumerate(sample_prompts):
        print(f"Processing prompt {i+1}/{len(sample_prompts)}: {prompt[:50]}...")
        
        # Get visualization data
        vis_data = setup_visualization_data(model_wrapper, crosscoder_adapter, prompt)
        
        # Extract feature-specific data
        feature_data = {
            "text": vis_data["text"],
            "token_strings": vis_data["token_strings"],
            "feature_activation": vis_data["feature_activations"][0, feature_idx].cpu().numpy().tolist(),
        }
        
        all_feature_data.append(feature_data)
        
        # Save the individual prompt data
        with open(os.path.join(feature_dir, f"prompt_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(feature_data, f, indent=2)
    
    # Get feature importance
    importance = get_feature_importance(crosscoder_adapter, feature_idx)
    
    # Create feature summary data
    feature_summary = {
        "feature_index": feature_idx,
        "importance": importance.cpu().numpy().tolist(),
        "samples": all_feature_data
    }
    
    # Save the summary data
    with open(os.path.join(feature_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2)
    
    # Create HTML visualization
    html_file = os.path.join(output_dir, f"feature_{feature_idx}.html")
    
    # Create a simple config for visualization
    config = data_config_classes.SequencesConfig(
        results_dir=feature_dir,
        max_context_len=64,
        max_examples=len(sample_prompts),
        batch_size=1,
        device="cpu"
    )
    
    # Generate HTML
    html_content = html_fns.get_html_feature_page(
        feature_summary,
        config,
        page_title=f"Feature {feature_idx} Visualization"
    )
    
    # Save HTML to file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Created visualization for feature {feature_idx} at {html_file}")
    
    return html_file

def visualize_top_features(
    model_wrapper,
    crosscoder_adapter,
    num_features: int,
    sample_prompts: List[str],
    output_dir: str,
    sae_vis_path: str
):
    """
    Generate visualizations for the top features by importance
    
    Args:
        model_wrapper: Model wrapper adapter
        crosscoder_adapter: CrossCoder adapter
        num_features: Number of top features to visualize
        sample_prompts: List of text prompts to use as examples
        output_dir: Directory to save visualizations
        sae_vis_path: Path to the sae_vis library
    
    Returns:
        Dictionary mapping feature indices to visualization HTML files
    """
    # Import sae_vis modules
    sae_vis_modules = import_sae_vis(sae_vis_path)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate feature importance for all features
    importances = []
    num_total_features = crosscoder_adapter.W_dec.shape[0]
    
    print(f"Calculating importance for {num_total_features} features...")
    for i in range(num_total_features):
        importance = get_feature_importance(crosscoder_adapter, i)
        importances.append((i, importance.sum().item()))
    
    # Sort features by importance
    sorted_features = sorted(importances, key=lambda x: x[1], reverse=True)
    
    # Select top features
    top_features = sorted_features[:num_features]
    
    print(f"Selected top {num_features} features by importance")
    
    # Create visualizations for top features
    visualization_files = {}
    
    for i, (feature_idx, importance) in enumerate(top_features):
        print(f"Creating visualization {i+1}/{num_features} for feature {feature_idx} (importance: {importance:.4f})")
        html_file = prepare_feature_visualization(
            model_wrapper, 
            crosscoder_adapter,
            feature_idx,
            sample_prompts,
            output_dir,
            sae_vis_modules
        )
        
        visualization_files[feature_idx] = html_file
    
    # Create index HTML file
    index_file = os.path.join(output_dir, "index.html")
    
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("<html><head><title>CrossCoder Feature Visualizations</title></head><body>\n")
        f.write("<h1>CrossCoder Feature Visualizations</h1>\n")
        f.write("<ul>\n")
        
        for feature_idx, html_file in visualization_files.items():
            html_filename = os.path.basename(html_file)
            f.write(f'<li><a href="{html_filename}">Feature {feature_idx}</a> (Importance: {dict(top_features)[feature_idx]:.4f})</li>\n')
        
        f.write("</ul></body></html>")
    
    print(f"Created index file at {index_file}")
    
    return visualization_files 
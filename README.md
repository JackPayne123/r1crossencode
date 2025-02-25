# CrossCoder Visualization for HuggingFace Models

This project provides an adapter that enables visualizing features of CrossCoder models trained with HuggingFace model activations, using the sae_vis visualization library.

## Overview

The [sae_vis library](https://github.com/anthropics/sae-vis-crosscoder) was originally designed to work with HookedTransformer-based models, while many researchers implement CrossCoders with HuggingFace models. This adapter bridges that gap, allowing you to:

1. Load a HuggingFace model and wrap it for activation inspection
2. Load a CrossCoder trained on that model's activations
3. Generate feature visualizations similar to those in the original sae_vis library

## Project Structure

- `sae_vis_adapter/`: The adapter package
  - `adapter.py`: CrossCoder adapter classes
  - `transformer_lens_adapter.py`: TransformerLens adapter for HuggingFace models
  - `observable_model.py`: Wrapper for inspecting HuggingFace model activations
  - `utils.py`: Utility functions
  - `visualization.py`: Feature visualization functions
  - `main.py`: Command-line script for generating visualizations

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Access to a trained CrossCoder checkpoint
- Access to the sae_vis library (from the sae_vis-crosscoder-vis repository)

### Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r sae_vis_adapter/requirements.txt
```

### Usage

#### Command-line Usage

The simplest way to generate visualizations is using the command-line script:

```bash
python -m sae_vis_adapter.main \
    --model_path gpt2 \                          # HF model name or path
    --crosscoder_path path/to/checkpoint.pt \    # CrossCoder checkpoint
    --sae_vis_path sae_vis-crosscoder-vis/sae_vis \ # Path to sae_vis library
    --output_dir visualizations \                # Output directory
    --layer 8 \                                  # Layer to visualize
    --hook_type attention_output \               # Type of hook
    --num_features 10                            # Number of features to visualize
```

#### Python API Usage

For more control, you can use the Python API:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_vis_adapter import (
    ObservableModel,
    CrossCoderAdapter, 
    CrossCoderConfig,
    TransformerLensWrapperAdapter,
    discover_hook_point_for_layer,
    visualize_top_features
)

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
observable_model = ObservableModel(model, tokenizer)

# Find a hook point
hook_point = discover_hook_point_for_layer(observable_model, 8, "attention_output")
model_wrapper = TransformerLensWrapperAdapter(observable_model, hook_point)

# Load CrossCoder
from your_module import CrossCoder  # Your HF-based CrossCoder implementation
checkpoint = torch.load("path/to/checkpoint.pt", map_location="cpu")
state_dict = checkpoint["model_state_dict"]

# Infer dimensions
W_enc = state_dict["W_enc"]
d_model = W_enc.shape[1]
d_hidden = W_enc.shape[2]

# Initialize CrossCoder
hf_crosscoder = CrossCoder(d_in=d_model, d_hidden=d_hidden)
hf_crosscoder.load_state_dict(state_dict)

# Create adapter
config = CrossCoderConfig(d_in=d_model, d_hidden=d_hidden)
crosscoder_adapter = CrossCoderAdapter(hf_crosscoder, config)

# Create visualizations
sample_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we work."
]

visualization_files = visualize_top_features(
    model_wrapper,
    crosscoder_adapter,
    num_features=10,
    sample_prompts=sample_prompts,
    output_dir="visualizations",
    sae_vis_path="sae_vis-crosscoder-vis/sae_vis"
)
```

## Customization

You can customize the adapter for your specific model by:

1. Implementing your own CrossCoder class compatible with HuggingFace models
2. Modifying the hook point discovery for your model's naming convention
3. Customizing the visualization parameters

## Results

The adapter will generate HTML visualizations of CrossCoder features, allowing you to:

- View the most important features based on decoder weights
- See how features activate across different inputs
- Analyze the token-level activation patterns

## License

[MIT License](LICENSE)

## Acknowledgements

This project builds upon:
- [sae_vis](https://github.com/anthropics/sae-vis-crosscoder) - The original visualization library for CrossCoder
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - For the model architecture

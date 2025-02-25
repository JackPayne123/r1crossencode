# CrossCoder Visualization Adapter

This adapter enables the visualization of HuggingFace-based CrossCoder models using the sae_vis visualization library. The adapter provides a bridge between HuggingFace models and the sae_vis library, which was originally designed for HookedTransformer-based models.

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from sae_vis_adapter import (
    ObservableModel,
    CrossCoderAdapter, 
    CrossCoderConfig,
    TransformerLensWrapperAdapter,
    discover_hook_point_for_layer,
    visualize_top_features
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load HuggingFace model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create ObservableModel wrapper
observable_model = ObservableModel(model, tokenizer)

# Discover a hook point for visualization
hook_point = discover_hook_point_for_layer(observable_model, layer_num=8, hook_type="attention_output")

# Create transformer lens wrapper
model_wrapper = TransformerLensWrapperAdapter(observable_model, hook_point)

# Load CrossCoder
crosscoder_checkpoint = "path/to/crosscoder_checkpoint.pt"
checkpoint = torch.load(crosscoder_checkpoint, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

# Infer dimensions from the checkpoint
W_enc = state_dict["W_enc"]
d_model = W_enc.shape[1]
d_hidden = W_enc.shape[2]

# Create HuggingFace CrossCoder (assume you have your own implementation)
from your_module import CrossCoder
hf_crosscoder = CrossCoder(d_in=d_model, d_hidden=d_hidden)
hf_crosscoder.load_state_dict(state_dict)

# Create adapter
config = CrossCoderConfig(d_in=d_model, d_hidden=d_hidden)
crosscoder_adapter = CrossCoderAdapter(hf_crosscoder, config)

# Create visualizations
sample_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "I think that artificial intelligence is going to transform how we live and work in the next decade."
]

visualization_files = visualize_top_features(
    model_wrapper,
    crosscoder_adapter,
    num_features=10,
    sample_prompts=sample_prompts,
    output_dir="visualizations",
    sae_vis_path="path/to/sae_vis"
)
```

### Using the Command Line

For quick usage, you can also use the provided command line script:

```bash
python -m sae_vis_adapter.main \
    --model_path gpt2 \
    --crosscoder_path path/to/crosscoder_checkpoint.pt \
    --sae_vis_path path/to/sae_vis \
    --output_dir visualizations \
    --layer 8 \
    --hook_type attention_output \
    --num_features 10
```

## Structure

The adapter consists of several key components:

1. `CrossCoderAdapter`: Adapts the HuggingFace CrossCoder to be compatible with sae_vis
2. `TransformerLensWrapperAdapter`: Mimics the TransformerLensWrapper from sae_vis for HuggingFace models
3. `ObservableModel`: Provides a way to hook into the internal activations of HuggingFace models
4. Visualization utilities: Tools for generating feature visualizations

## Customization

You can customize the adapter for your specific model by:

1. Modifying the `discover_hook_point_for_layer` function to support your model's hook point naming convention
2. Adjusting the `ObservableModel` class to handle your model's specific architecture
3. Modifying the visualization parameters in `prepare_feature_visualization`

## Requirements

- PyTorch
- Transformers
- einops
- numpy

## License

[MIT License](LICENSE) 
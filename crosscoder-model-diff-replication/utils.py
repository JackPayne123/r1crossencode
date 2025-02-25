# %%
import datetime
import os
from IPython import get_ipython
from google.colab import drive
from pathlib import Path
import random

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Import stuff
import einops
import json
import argparse

from datasets import load_dataset
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import wandb
import pprint

def to_numpy(tensor):
    """Convert a tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, (int, float)):
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported type for to_numpy conversion: {type(tensor)}")

class ObservableModel:
    """
    A wrapper for HuggingFace models that allows for activation capture and intervention.
    Replaces TransformerLens functionality with native PyTorch hooks.
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        quantization_config: Optional[Any] = None,
    ):
        self.dtype = dtype
        self.device = device
        
        # Configure model initialization properly
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": device,
            "trust_remote_code": trust_remote_code,  # Required for Qwen models
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            
        try:
            # First load the config to modify it
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Enable caching for activation capture
            config.use_cache = True
            if hasattr(config, "cache_implementation"):
                config.cache_implementation = "standard"  # Use standard caching
            
            # Add config to model kwargs
            model_kwargs["config"] = config
            
            print(f"\nInitializing {model_name} with config:")
            pprint.pprint(model_kwargs)
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=trust_remote_code  # Required for Qwen models
            )
            
            # Ensure padding token is set for Qwen models
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.cfg = self._model.config
            
            # Cache for storing activations during forward pass
            self.activation_cache = {}
            
            print(f"\nModel initialized successfully!")
            print(f"Model type: {type(self._model).__name__}")
            print(f"Hidden size: {self.cfg.hidden_size}")
            print(f"Number of layers: {self.cfg.num_hidden_layers}")
            print(f"Caching enabled: {self.cfg.use_cache}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def print_available_hook_points(self):
        """Print the available hook points in the model structure"""
        def get_named_modules(model, prefix=""):
            for name, module in model.named_modules():
                if name:  # Skip empty names
                    print(f"- {name}")
                    
        get_named_modules(self._model)
        
    def _find_module(self, hook_point: str) -> nn.Module:
        """Finds a module given its name in dot notation."""
        try:
            submodules = hook_point.split(".")
            module = self._model
            while submodules:
                module = getattr(module, submodules.pop(0))
            return module
        except Exception as e:
            raise ValueError(f"Could not find module {hook_point}: {str(e)}")
    
    def run_with_cache(
        self, 
        input_ids: torch.Tensor,
        names_filter: Optional[str] = None
    ) -> tuple[Any, Dict[str, torch.Tensor]]:
        """
        Run the model while caching activations at specified points.
        Similar to TransformerLens's run_with_cache but for HF models.
        """
        self.activation_cache = {}
        
        def cache_hook(name: str):
            def hook(mod, inputs, outputs):
                # For LayerNorm and RMSNorm, we want to cache the normalized activations
                if isinstance(mod, (nn.LayerNorm, nn.RMSNorm)) or "input_layernorm" in name:
                    if isinstance(outputs, tuple):
                        self.activation_cache[name] = outputs[0].detach()
                    else:
                        self.activation_cache[name] = outputs.detach()
                else:
                    if isinstance(outputs, tuple):
                        self.activation_cache[name] = outputs[0].detach()
                    else:
                        self.activation_cache[name] = outputs.detach()
                return outputs
            return hook
        
        handles = []
        if names_filter:
            try:
                module = self._find_module(names_filter)
                handles.append(module.register_forward_hook(cache_hook(names_filter)))
            except Exception as e:
                print(f"Error registering hook for {names_filter}: {str(e)}")
                raise
            
        try:
            with torch.no_grad():
                outputs = self._model.forward(
                    input_ids,
                    use_cache=True,  # Explicitly enable caching for activation capture
                    output_hidden_states=True  # Ensure we get all hidden states
                )
        finally:
            for handle in handles:
                handle.remove()
                
        return outputs, self.activation_cache
    
    @property
    def cfg(self):
        """Access to model config, similar to TransformerLens."""
        return self._model.config
    
    @cfg.setter
    def cfg(self, value):
        self._model.config = value

from jaxtyping import Float
#from transformer_lens.hook_points import HookPoint

from functools import partial

from IPython.display import HTML

#from transformer_lens.utils import to_numpy
import pandas as pd

from html import escape
import colorsys

import plotly.graph_objects as go

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
    "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
    "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth"
}

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(to_numpy(tensor), color_continuous_midpoint=0.0,labels={"x":xaxis, "y":yaxis}, **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    fig.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    if return_fig:
        return fig
    fig.show(renderer)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def bar(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.bar(
        y=to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template="simple_white",
        **kwargs).show(renderer)

def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))

# crosscoder stuff

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg    

def prepare_tokenized_dataset(texts, tokenizer_A, tokenizer_B, max_length=2048):
    """Pre-tokenize dataset with both tokenizers and ensure they're aligned"""
    try:
        # Check for cached tokenized data in Drive first
        drive_dir = Path("/content/drive/MyDrive/crosscoder_data/tokenized")
        drive_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filenames based on tokenizer names and dataset composition
        model_A_name = tokenizer_A.name_or_path.replace('/', '_')
        model_B_name = tokenizer_B.name_or_path.replace('/', '_')
        
        # Include dataset size in cache filename
        cache_identifier = f"{len(texts)}_samples"
        tokens_A_cache = drive_dir / f"tokens_{model_A_name}_{cache_identifier}.pt"
        tokens_B_cache = drive_dir / f"tokens_{model_B_name}_{cache_identifier}.pt"
        
        # Initialize tokens
        tokens_A = None
        tokens_B = None
        
        # Try to load Model A tokens from cache
        if tokens_A_cache.exists():
            print(f"Loading cached tokens for Model A ({model_A_name})...")
            try:
                tokens_A = torch.load(tokens_A_cache)
                print(f"Successfully loaded Model A tokens: {tokens_A.shape}")
            except Exception as e:
                print(f"Error loading Model A tokens: {str(e)}")
                tokens_A = None
                
        # Try to load Model B tokens from cache
        if tokens_B_cache.exists():
            print(f"Loading cached tokens for Model B ({model_B_name})...")
            try:
                tokens_B = torch.load(tokens_B_cache)
                print(f"Successfully loaded Model B tokens: {tokens_B.shape}")
            except Exception as e:
                print(f"Error loading Model B tokens: {str(e)}")
                tokens_B = None
                
        # Tokenize Model A if needed
        if tokens_A is None:
            print(f"\nTokenizing with Model A's tokenizer ({model_A_name})...")
            batch_size = 1000
            num_batches = (len(texts) + batch_size - 1) // batch_size
            tokens_A_list = []
            
            for i in tqdm.tqdm(range(num_batches), desc="Model A tokenization"):
                batch_texts = texts[i * batch_size:(i + 1) * batch_size]
                batch_tokens = tokenizer_A(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).input_ids
                tokens_A_list.append(batch_tokens)
            
            tokens_A = torch.cat(tokens_A_list, dim=0)
            
            # Save Model A tokens
            print(f"Saving Model A tokens to {tokens_A_cache}")
            torch.save(tokens_A, tokens_A_cache)
            
        # Tokenize Model B if needed
        if tokens_B is None:
            print(f"\nTokenizing with Model B's tokenizer ({model_B_name})...")
            batch_size = 1000
            num_batches = (len(texts) + batch_size - 1) // batch_size
            tokens_B_list = []
            
            for i in tqdm.tqdm(range(num_batches), desc="Model B tokenization"):
                batch_texts = texts[i * batch_size:(i + 1) * batch_size]
                batch_tokens = tokenizer_B(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).input_ids
                tokens_B_list.append(batch_tokens)
            
            tokens_B = torch.cat(tokens_B_list, dim=0)
            
            # Save Model B tokens
            print(f"Saving Model B tokens to {tokens_B_cache}")
            torch.save(tokens_B, tokens_B_cache)
            
        # Ensure we have the same number of sequences
        min_seqs = min(tokens_A.shape[0], tokens_B.shape[0])
        tokens_A = tokens_A[:min_seqs]
        tokens_B = tokens_B[:min_seqs]
        
        # Save dataset info
        info_file = drive_dir / f"tokenization_info_{cache_identifier}.json"
        info = {
            "total_samples": len(texts),
            "max_length": max_length,
            "model_A": model_A_name,
            "model_B": model_B_name,
            "tokens_A_shape": list(tokens_A.shape),
            "tokens_B_shape": list(tokens_B.shape),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
            
        print(f"\nFinal tokenized shapes:")
        print(f"Model A: {tokens_A.shape}")
        print(f"Model B: {tokens_B.shape}")
        
        return {
            "tokens_A": tokens_A,
            "tokens_B": tokens_B
        }
        
    except Exception as e:
        print(f"Error tokenizing dataset: {str(e)}")
        raise

def format_chat_for_qwen(conversation):
    """
    Formats LMSys chat data to match Qwen's expected chat format.
    This ensures the model will process chat data similarly to how it was trained.
    """
    formatted_chat = ""
    for message in conversation:
        role = message["role"]
        content = message["content"]

        # Format based on Qwen's chat template
        if role == "user":
            formatted_chat += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_chat += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    return formatted_chat

def format_openthoughts_for_qwen(sample):
    """Format OpenThoughts examples for Qwen's chat format"""
    try:
        # Extract system prompt and conversations
        system = sample.get('system', '').strip()
        conversations = sample.get('conversations', [])
        
        # Skip if no conversations
        if not conversations:
            return None
            
        # Format as Qwen chat with system prompt
        formatted_text = ""
        
        # Add system prompt if present
        if system:
            formatted_text += f"<|im_start|>system\n{system}<|im_end|>\n"
        
        # Add each conversation turn
        for conv in conversations:
            # Each conversation item is a dict with 'from' and 'value' keys
            role = conv.get('from', '').strip()
            content = conv.get('value', '').strip()
            
            # Map 'from' to Qwen's role format
            if role == 'user':
                qwen_role = 'user'
            elif role == 'assistant':
                qwen_role = 'assistant'
            else:
                continue  # Skip unknown roles
            
            if content:
                formatted_text += f"<|im_start|>{qwen_role}\n{content}<|im_end|>\n"
        
        # Validate the formatting
        if not validate_openthoughts_sample(formatted_text):
            return None
            
        return formatted_text
        
    except Exception as e:
        print(f"Error formatting OpenThoughts sample: {str(e)}")
        return None

def validate_openthoughts_sample(formatted_text: str) -> bool:
    """Validates that an OpenThoughts sample matches the expected format"""
    try:
        # Check if there's any content
        if not formatted_text.strip():
            return False
        
        # Split into turns
        turns = formatted_text.split("<|im_start|>")
        turns = [t for t in turns if t.strip()]  # Remove empty turns
        
        if not turns:
            return False
        
        # Must have at least a user and assistant turn
        has_user = False
        has_assistant = False
        
        for turn in turns:
            if not "<|im_end|>" in turn:
                return False
                
            if turn.startswith("user\n"):
                has_user = True
            elif turn.startswith("assistant\n"):
                has_assistant = True
                
        # Must have at least one user and one assistant message
        return has_user and has_assistant
        
    except Exception as e:
        print(f"Error validating OpenThoughts sample: {str(e)}")
        return False

def load_raw_dataset():
    """Creates a balanced dataset with proper caching and shuffling"""
    try:
        # Setup Drive caching for final combined dataset
        drive_dir = Path("/content/drive/MyDrive/crosscoder_data")
        drive_dir.mkdir(parents=True, exist_ok=True)
        raw_cache_path = drive_dir / "combined_raw_data.txt"
        
        # Try to load cached combined data first
        if raw_cache_path.exists():
            print("Loading cached combined dataset from Drive...")
            try:
                with open(raw_cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts = content.split("\n\n")
                    texts = [t.strip() for t in texts if t.strip()]
                print(f"Loaded {len(texts)} cached text sequences")
                return texts
            except Exception as e:
                print(f"Error reading cached data: {str(e)}")
                print("Recreating dataset...")
        
        print("Creating new balanced dataset...")
        target_samples = 70_000  # Per source
        all_samples = []
        
        # Load and process Pile data (Parquet format)
        print("\nLoading Pile data...")
        try:
            pile = load_dataset(
                "monology/pile-uncopyrighted", 
                split="train", 
                streaming=True
            )
            pile_samples = []
            pile_iter = iter(pile)
            
            with tqdm.tqdm(total=target_samples) as pbar:
                pile_count = 0
                while pile_count < target_samples:
                    try:
                        sample = next(pile_iter)
                        pile_samples.append({
                            "text": sample['text'],
                            "type": "pile"
                        })
                        pile_count += 1
                        pbar.update(1)
                    except StopIteration:
                        break
            all_samples.extend(pile_samples)
            print(f"Collected {len(pile_samples)} Pile samples")
        except Exception as e:
            print(f"Error loading Pile data: {str(e)}")
            pile_samples = []
        
        # Load and process LMSys data (Parquet format)
        print("\nLoading LMSys chat data...")
        try:
            lmsys = load_dataset(
                "lmsys/lmsys-chat-1m", 
                split="train", 
                streaming=True
            )
            lmsys_samples = []
            lmsys_iter = iter(lmsys)
            
            with tqdm.tqdm(total=target_samples) as pbar:
                lmsys_count = 0
                while lmsys_count < target_samples:
                    try:
                        sample = next(lmsys_iter)
                        chat_text = format_chat_for_qwen(sample["conversation"])
                        lmsys_samples.append({
                            "text": chat_text,
                            "type": "lmsys"
                        })
                        lmsys_count += 1
                        pbar.update(1)
                    except StopIteration:
                        break
            all_samples.extend(lmsys_samples)
            print(f"Collected {len(lmsys_samples)} LMSys samples")
        except Exception as e:
            print(f"Error loading LMSys data: {str(e)}")
            lmsys_samples = []
        
        # Load and process OpenThoughts data (Parquet format)
        print("\nLoading OpenThoughts data...")
        try:
            openthoughts = load_dataset(
                "open-thoughts/OpenThoughts-114k",
                split="train"
            )
            print(f"Loaded OpenThoughts with {len(openthoughts)} examples")
            
            # Print first example structure
            print("\nExample OpenThoughts structure:")
            example = openthoughts[0]
            print("System:", example.get('system', ''))
            print("First conversation turn:", example.get('conversations', [])[0] if example.get('conversations') else None)
            
            openthoughts_samples = []
            with tqdm.tqdm(total=target_samples) as pbar:
                indices = torch.randperm(len(openthoughts)).tolist()
                
                for idx in indices:
                    if len(openthoughts_samples) >= target_samples:
                        break
                    sample = openthoughts[idx]
                    formatted_text = format_openthoughts_for_qwen(sample)
                    
                    if formatted_text is not None:
                        openthoughts_samples.append({
                            "text": formatted_text,
                            "type": "openthoughts"
                        })
                        pbar.update(1)
                        
                        # Print first successful example
                        if len(openthoughts_samples) == 1:
                            print("\nFirst formatted OpenThoughts example:")
                            print(formatted_text[:500] + "...")
                    elif len(openthoughts_samples) < 5:  # Print first few failures
                        print(f"\nFailed to format sample {idx}:")
                        print("System:", sample.get('system', ''))
                        print("Conversations:", sample.get('conversations', []))
            
            all_samples.extend(openthoughts_samples)
            print(f"Collected {len(openthoughts_samples)} OpenThoughts samples")
            
        except Exception as e:
            print(f"Error loading OpenThoughts data: {str(e)}")
            print("Detailed error info:")
            import traceback
            traceback.print_exc()
            openthoughts_samples = []
        
        # Shuffle all samples
        print("\nShuffling combined dataset...")
        np.random.shuffle(all_samples)
        
        # Extract text only for final dataset
        texts = [sample["text"] for sample in all_samples]
        
        # Print dataset composition
        type_counts = {}
        for sample in all_samples:
            type_counts[sample["type"]] = type_counts.get(sample["type"], 0) + 1
            
        print("\nDataset composition:")
        for data_type, count in type_counts.items():
            print(f"{data_type.capitalize()} samples: {count}")
        print(f"Total samples: {len(texts)}")
        
        # Save combined dataset to Drive
        print("\nSaving combined dataset to Drive...")
        try:
            with open(raw_cache_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts))
            print(f"Successfully saved combined dataset to {raw_cache_path}")
        except Exception as e:
            print(f"Error saving combined dataset: {str(e)}")
        
        return texts
        
    except Exception as e:
        print(f"Error in load_raw_dataset: {str(e)}")
        raise

def load_pile_lmsys_mixed_tokens(model_A=None, model_B=None):
    """Load or create tokenized dataset for model comparison."""
    try:
        # Load raw text data
        raw_texts = load_raw_dataset()
        
        if model_A is None or model_B is None:
            print("Models not provided, returning raw text")
            return {"raw_texts": raw_texts}
        
        # Now tokenize with both models
        print("Tokenizing data with both models...")
        tokenized_data = prepare_tokenized_dataset(
            raw_texts,
            model_A.tokenizer,
            model_B.tokenizer,
            max_length=2048
        )
        
        return tokenized_data
        
    except Exception as e:
        print(f"Error in load_pile_lmsys_mixed_tokens: {str(e)}")
        raise

def splice_act_hook(mod, inputs, outputs, spliced_act):
    """Hook function to splice in reconstructed activations"""
    # Drop BOS token and replace remaining activations
    outputs[:, 1:, :] = spliced_act
    return outputs
    
def zero_ablation_hook(mod, inputs, outputs):
    """Hook function to zero out activations"""
    outputs[:] = 0
    return outputs

def get_ce_recovered_metrics(tokens, model_A, model_B, cross_coder):
    """
    Calculate CE recovery metrics for comparing model activations through CrossCoder.
    
    Args:
        tokens: Input tokens tensor [batch_size, seq_len]
        model_A: First ObservableModel instance
        model_B: Second ObservableModel instance
        cross_coder: Trained CrossCoder model
        
    Returns:
        Dictionary of CE recovery metrics
    """
    try:
        # Ensure CrossCoder is in float32
        cross_coder = cross_coder.float()
        
        # Prepare labels for loss calculation (shift tokens by 1)
        labels = tokens[:, 1:].contiguous()
        input_tokens = tokens[:, :-1].contiguous()
        
        # Get clean loss (no hooks)
        with torch.no_grad():
            outputs_A = model_A._model(input_tokens, labels=labels)
            outputs_B = model_B._model(input_tokens, labels=labels)
            ce_clean_A = outputs_A.loss
            ce_clean_B = outputs_B.loss
            
            print(f"\nClean loss check:")
            print(f"CE clean A: {ce_clean_A.item()}")
            print(f"CE clean B: {ce_clean_B.item()}")

        # Get zero ablation loss
        with torch.no_grad():
            # Register zero ablation hook
            hook_point = cross_coder.cfg["hook_point"]
            module_A = model_A._find_module(hook_point)
            module_B = model_B._find_module(hook_point)
            
            handle_A = module_A.register_forward_hook(zero_ablation_hook)
            handle_B = module_B.register_forward_hook(zero_ablation_hook)
            
            outputs_A = model_A._model(input_tokens, labels=labels)
            outputs_B = model_B._model(input_tokens, labels=labels)
            ce_zero_abl_A = outputs_A.loss
            ce_zero_abl_B = outputs_B.loss
            
            print(f"\nZero ablation loss check:")
            print(f"CE zero A: {ce_zero_abl_A.item()}")
            print(f"CE zero B: {ce_zero_abl_B.item()}")
            
            # Remove hooks
            handle_A.remove()
            handle_B.remove()

        # Get activations for splicing
        with torch.no_grad():
            _, cache_A = model_A.run_with_cache(input_tokens, names_filter=hook_point)
            _, cache_B = model_B.run_with_cache(input_tokens, names_filter=hook_point)
            
            # Convert activations to float32
            resid_act_A = cache_A[hook_point].float()
            resid_act_B = cache_B[hook_point].float()

            # Stack and prepare for CrossCoder
            cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=1)  # [batch, 2, seq_len, hidden]
            cross_coder_input = cross_coder_input[:, :, 1:, :]  # Drop BOS
            
            # Reshape for CrossCoder
            batch_size, n_models, seq_len, hidden = cross_coder_input.shape
            cross_coder_input = cross_coder_input.reshape(batch_size * seq_len, n_models, hidden)

            print(f"\nDtype check before CrossCoder:")
            print(f"cross_coder_input dtype: {cross_coder_input.dtype}")
            print(f"W_enc dtype: {cross_coder.W_enc.dtype}")
            print(f"W_dec dtype: {cross_coder.W_dec.dtype}")

            # Get reconstructed activations
            cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
            cross_coder_output = cross_coder_output.reshape(batch_size, seq_len, n_models, hidden)
            cross_coder_output = cross_coder_output.transpose(1, 2)  # [batch, n_models, seq_len, hidden]
            
            cross_coder_output_A = cross_coder_output[:, 0]  # [batch, seq_len, hidden]
            cross_coder_output_B = cross_coder_output[:, 1]  # [batch, seq_len, hidden]

        # Get spliced loss
        with torch.no_grad():
            # Register splicing hooks
            handle_A = module_A.register_forward_hook(
                lambda mod, inp, out: splice_act_hook(mod, inp, out, cross_coder_output_A)
            )
            handle_B = module_B.register_forward_hook(
                lambda mod, inp, out: splice_act_hook(mod, inp, out, cross_coder_output_B)
            )
            
            outputs_A = model_A._model(input_tokens, labels=labels)
            outputs_B = model_B._model(input_tokens, labels=labels)
            ce_loss_spliced_A = outputs_A.loss
            ce_loss_spliced_B = outputs_B.loss
            
            print(f"\nSpliced loss check:")
            print(f"CE spliced A: {ce_loss_spliced_A.item()}")
            print(f"CE spliced B: {ce_loss_spliced_B.item()}")
            
            # Remove hooks
            handle_A.remove()
            handle_B.remove()

        # Compute CE recovery percentages
        ce_recovered_A = 1 - ((ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A))
        ce_recovered_B = 1 - ((ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B))

        metrics = {
            "ce_loss_spliced_A": ce_loss_spliced_A.item(),
            "ce_loss_spliced_B": ce_loss_spliced_B.item(),
            "ce_clean_A": ce_clean_A.item(),
            "ce_clean_B": ce_clean_B.item(),
            "ce_zero_abl_A": ce_zero_abl_A.item(),
            "ce_zero_abl_B": ce_zero_abl_B.item(),
            "ce_diff_A": (ce_loss_spliced_A - ce_clean_A).item(),
            "ce_diff_B": (ce_loss_spliced_B - ce_clean_B).item(),
            "ce_recovered_A": ce_recovered_A.item(),
            "ce_recovered_B": ce_recovered_B.item(),
        }
        return metrics
        
    except Exception as e:
        print(f"Error calculating CE metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
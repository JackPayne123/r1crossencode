from typing import Optional, Union, Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass

@dataclass
class CrossCoderConfig:
    """Class for storing configuration parameters for the CrossCoder adapter"""
    d_in: int
    d_hidden: int
    dict_mult: Optional[int] = None
    apply_b_dec_to_input: bool = False
    l1_coeff: float = 3e-4

    def __post_init__(self):
        if self.dict_mult is None and isinstance(self.d_hidden, int):
            self.dict_mult = self.d_hidden // self.d_in


class CrossCoderAdapter(nn.Module):
    """
    Adapter class to make HuggingFace-based CrossCoder compatible with sae_vis
    Designed to be a drop-in replacement for sae_vis.model_fns.CrossCoder
    """
    def __init__(self, hf_crosscoder=None, config=None):
        """
        Initialize the adapter with a HuggingFace-based CrossCoder.
        
        Args:
            hf_crosscoder: The HuggingFace-based CrossCoder instance or None
            config: Configuration for the adapter
        """
        super().__init__()
        if hf_crosscoder is not None:
            self.hf_crosscoder = hf_crosscoder
            self.cfg = config
            
            # Reference the weights from the HuggingFace CrossCoder
            self.W_enc = hf_crosscoder.W_enc
            self.W_dec = hf_crosscoder.W_dec
            self.b_enc = hf_crosscoder.b_enc
            self.b_dec = hf_crosscoder.b_dec
            
            # For compatibility with sae_vis, store cfg as attributes
            if hasattr(hf_crosscoder, "cfg") and isinstance(hf_crosscoder.cfg, dict):
                self.hook_point = hf_crosscoder.cfg.get("hook_point", None)
                self.l1_coeff = hf_crosscoder.cfg.get("l1_coeff", 3e-4)
            else:
                self.hook_point = None
                self.l1_coeff = config.l1_coeff if config else 3e-4
        else:
            # If no crosscoder provided, this should be initialized manually later
            self.W_enc = None
            self.W_dec = None
            self.b_enc = None
            self.b_dec = None
            self.cfg = config
            self.hook_point = None
            self.l1_coeff = config.l1_coeff if config else 3e-4
    
    def load_state_dict_from_original(self, original_state_dict):
        """
        Load state dict from original CrossCoder implementation
        
        Args:
            original_state_dict: State dict from original CrossCoder
        """
        self.load_state_dict(original_state_dict)
        return self
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for compatibility with sae_vis.
        
        Args:
            x: Input tensor of shape [batch, n_models, d_model]
        
        Returns:
            Tuple of (loss, reconstructed input, activations, l2_loss, l1_loss)
        """
        # Apply encoder
        x_cent = x
        x_enc = einops.einsum(
            x_cent,
            self.W_enc,
            "... n_layers d_model, n_layers d_model d_hidden -> ... d_hidden",
        )
        acts = F.relu(x_enc + self.b_enc)
        
        # Apply decoder
        x_reconstruct = einops.einsum(
            acts,
            self.W_dec,
            "... d_hidden, d_hidden n_layers d_model -> ... n_layers d_model",
        ) + self.b_dec
        
        # Calculate losses (similar to sae_vis CrossCoder)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_layers d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()
        
        # Calculate L1 loss
        decoder_norms = self.W_dec.norm(dim=-1)
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_layers -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        
        # Combined loss
        loss = l2_loss + l1_loss * self.l1_coeff
        
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    def encode(self, x: torch.Tensor, apply_relu: bool = True) -> torch.Tensor:
        """
        Encode input tensor to feature space (for compatibility with sae_vis)
        
        Args:
            x: Input tensor of shape [batch, n_models, d_model]
            apply_relu: Whether to apply ReLU to the encoded activations
            
        Returns:
            Encoded features
        """
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "... n_layers d_model, n_layers d_model d_hidden -> ... d_hidden",
        )
        if apply_relu:
            return F.relu(x_enc + self.b_enc)
        else:
            return x_enc + self.b_enc

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Decode activations back to input space (for compatibility with sae_vis)
        
        Args:
            acts: Activation tensor of shape [batch, d_hidden]
            
        Returns:
            Decoded input of shape [batch, n_models, d_model]
        """
        x_reconstruct = einops.einsum(
            acts,
            self.W_dec,
            "... d_hidden, d_hidden n_layers d_model -> ... n_layers d_model",
        ) + self.b_dec
        
        return x_reconstruct
    
    def to(self, *args, **kwargs):
        """Transfer the model to device/dtype"""
        if hasattr(self, 'hf_crosscoder') and self.hf_crosscoder is not None:
            self.hf_crosscoder.to(*args, **kwargs)
        else:
            if self.W_enc is not None:
                self.W_enc = self.W_enc.to(*args, **kwargs)
            if self.W_dec is not None:
                self.W_dec = self.W_dec.to(*args, **kwargs)
            if self.b_enc is not None:
                self.b_enc = self.b_enc.to(*args, **kwargs)
            if self.b_dec is not None:
                self.b_dec = self.b_dec.to(*args, **kwargs)
        return self
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict - in this case, we're already using the HF model's weights"""
        if hasattr(self, 'hf_crosscoder') and self.hf_crosscoder is not None:
            return self.hf_crosscoder.load_state_dict(state_dict, strict=strict)
        else:
            return super().load_state_dict(state_dict, strict=strict)
    
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        """Implement the sae_vis CrossCoder method for compatibility"""
        if hasattr(self, 'hf_crosscoder') and self.hf_crosscoder is not None:
            if self.hf_crosscoder.W_dec.grad is not None:
                W_dec_normed = self.hf_crosscoder.W_dec / self.hf_crosscoder.W_dec.norm(dim=-1, keepdim=True)
                W_dec_grad_proj = (self.hf_crosscoder.W_dec.grad * W_dec_normed).sum(
                    -1, keepdim=True
                ) * W_dec_normed
                self.hf_crosscoder.W_dec.grad -= W_dec_grad_proj
        else:
            if self.W_dec.grad is not None:
                W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
                W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                    -1, keepdim=True
                ) * W_dec_normed
                self.W_dec.grad -= W_dec_grad_proj


class ObservableModelAdapter:
    """
    Adapter class to make HuggingFace ObservableModel compatible with TransformerLensWrapper
    """
    def __init__(self, observable_model, hook_point: str):
        """
        Initialize the adapter with a HuggingFace ObservableModel.
        
        Args:
            observable_model: The ObservableModel instance
            hook_point: The hook point to use for getting activations
        """
        self.model = observable_model
        self.hook_point = hook_point
        self.tokenizer = observable_model.tokenizer
    
    @property
    def W_U(self):
        """Unembedding weights - used for logit lens"""
        # Depending on the model, get the output weights (lm_head or W_U equivalent)
        if hasattr(self.model._model, 'lm_head'):
            return self.model._model.lm_head.weight
        elif hasattr(self.model._model, 'output'):
            return self.model._model.output.weight
        # Fallback for other model architectures
        raise ValueError("Could not find W_U equivalent (lm_head) in the model")
    
    @property
    def W_out(self):
        """MLP output weights - needed if looking at MLP activations"""
        # This will need to be properly implemented based on your specific model architecture
        # For example, if using a specific layer's MLP output
        return None  # Return proper weights if needed
    
    def forward(self, tokens, return_logits=True):
        """
        Run the model forward pass, compatible with the TransformerLensWrapper interface
        
        Args:
            tokens: Input token IDs
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (logits, residual, activation) or (residual, activation)
        """
        with torch.no_grad():
            # Run the model with cache to get activations
            outputs, cache = self.model.run_with_cache(tokens, names_filter=self.hook_point)
            
            # Get the activation at the specified hook point
            activation = cache[self.hook_point]
            
            # For residual, we use the same activation - in a proper implementation
            # you might need to map to a different part of the model depending on architecture
            residual = activation
            
            if return_logits:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                return logits, residual, activation
            else:
                return residual, activation 
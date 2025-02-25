from typing import Optional, Union, Dict, List, Any, Tuple
import torch
import torch.nn as nn
import einops


class TransformerLensWrapperAdapter:
    """
    Adapter class to mimic the TransformerLensWrapper from sae_vis but work with HuggingFace models
    """
    
    def __init__(self, observable_model, hook_point: str):
        """
        Initialize the wrapper with a HuggingFace ObservableModel.
        
        Args:
            observable_model: The ObservableModel instance wrapping a HuggingFace model
            hook_point: The hook point to use for getting activations
        """
        self.model = observable_model
        self.hook_point = hook_point
        self.tokenizer = observable_model.tokenizer
        
        # For convenience, expose the device of the model
        self.device = next(observable_model._model.parameters()).device
    
    def get_activations(self, tokens: torch.Tensor, hook_point: Optional[str] = None) -> torch.Tensor:
        """
        Get activations from the specified hook point
        
        Args:
            tokens: Input token IDs
            hook_point: Override the default hook point
            
        Returns:
            Activations tensor
        """
        hook_to_use = hook_point or self.hook_point
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=hook_to_use)
            return cache[hook_to_use]
    
    def get_residual_stream(self, tokens: torch.Tensor, hook_point: Optional[str] = None) -> torch.Tensor:
        """
        Get residual stream activations (may be the same as regular activations depending on the model)
        
        Args:
            tokens: Input token IDs
            hook_point: Override the default hook point
            
        Returns:
            Residual stream tensor
        """
        # For many models, this might be the same as get_activations
        # For models with different residual structures, this would need to be implemented differently
        return self.get_activations(tokens, hook_point)
    
    def forward(self, tokens: torch.Tensor, return_logits: bool = True) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Run the model forward pass
        
        Args:
            tokens: Input token IDs
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (logits, residual, activation) or (residual, activation)
        """
        with torch.no_grad():
            # Run model and get activations
            outputs = self.model._model(tokens)
            
            # Get activations
            _, cache = self.model.run_with_cache(tokens, names_filter=self.hook_point)
            activations = cache[self.hook_point]
            
            # Get residual - in many cases, this might be the same as activations
            # depending on the model architecture
            residual = activations
            
            if return_logits:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                return logits, residual, activations
            else:
                return residual, activations
    
    def to_resid_dir(self, direction: torch.Tensor, hook_point: Optional[str] = None) -> torch.Tensor:
        """
        Convert a direction in activation space to a direction in residual space
        
        Args:
            direction: Direction tensor in activation space
            hook_point: Override the default hook point
            
        Returns:
            Direction tensor in residual space
        """
        # In many models, this might just be an identity transformation
        # as the activation and residual spaces could be the same
        return direction
    
    def generate(self, prompt: Union[str, torch.Tensor], max_new_tokens: int = 20) -> str:
        """
        Generate text from the model
        
        Args:
            prompt: Text prompt or token IDs
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated text
        """
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = prompt
            
        with torch.no_grad():
            output_ids = self.model._model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def to(self, device: torch.device) -> "TransformerLensWrapperAdapter":
        """
        Move the model to the specified device
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self, after moving to device
        """
        self.model._model.to(device)
        self.device = device
        return self 
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

class ObservableModel:
    """
    A wrapper around transformers models that allows for observing internal activations
    and applying interventions at various hook points
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the observable model with a HuggingFace model and tokenizer
        
        Args:
            model: The HuggingFace model to wrap
            tokenizer: The tokenizer for the model
        """
        self._model = model
        self.tokenizer = tokenizer
        self._hooks = {}
        self._hook_points = None
        
        # Identify available hook points
        self._identify_hook_points()
    
    def _identify_hook_points(self):
        """
        Identify available hook points in the model
        """
        self._hook_points = {}
        
        # Register hook points for each module
        for name, module in self._model.named_modules():
            # Skip the top module
            if name == "":
                continue
            
            # Add this module as a hook point
            self._hook_points[name] = module
    
    def get_available_hook_points(self) -> List[str]:
        """
        Get the list of available hook points
        
        Returns:
            List of hook point names
        """
        return list(self._hook_points.keys())
    
    def print_available_hook_points(self):
        """
        Print the list of available hook points
        """
        hook_points = self.get_available_hook_points()
        print(f"Available hook points ({len(hook_points)}):")
        for i, hp in enumerate(sorted(hook_points)):
            print(f"{i+1}. {hp}")
    
    def _hook_fn(self, name: str, module: nn.Module, input_tensor: Tuple[torch.Tensor], output_tensor: torch.Tensor):
        """
        Hook function that's called when a module with a registered hook is executed
        
        Args:
            name: Name of the hook point
            module: The module being executed
            input_tensor: Input tensor to the module
            output_tensor: Output tensor from the module
        """
        # Store the output in the cache
        if hasattr(self, "_cache"):
            self._cache[name] = output_tensor
        
        # Apply any interventions
        if name in self._interventions:
            intervention_fn = self._interventions[name]
            # Return the intervention result
            return intervention_fn(output_tensor)
        
        # Otherwise, return the original output
        return output_tensor
    
    def _setup_hooks(self, names_filter: Optional[Union[str, List[str]]] = None):
        """
        Set up hooks for caching activations
        
        Args:
            names_filter: Optional filter for which hook points to register. 
                          If None, all hook points are registered.
                          If a string, only hook points containing that string are registered.
                          If a list, only hook points in the list are registered.
        """
        # Clear any existing hooks
        self.remove_hooks()
        
        # Create the cache
        self._cache = {}
        self._interventions = {}
        
        # Determine which hook points to register
        hook_points = self.get_available_hook_points()
        if names_filter is not None:
            if isinstance(names_filter, str):
                hook_points = [hp for hp in hook_points if names_filter in hp]
            else:
                hook_points = [hp for hp in hook_points if hp in names_filter]
        
        # Register hooks
        for name in hook_points:
            module = self._hook_points[name]
            hook = module.register_forward_hook(
                lambda mod, inp, out, name=name: self._hook_fn(name, mod, inp, out)
            )
            self._hooks[name] = hook
    
    def remove_hooks(self):
        """
        Remove all registered hooks
        """
        for hook in self._hooks.values():
            hook.remove()
        self._hooks = {}
        self._cache = {}
        self._interventions = {}
    
    def set_intervention(self, hook_point: str, intervention_fn):
        """
        Set an intervention function for a hook point
        
        Args:
            hook_point: Name of the hook point
            intervention_fn: Function that takes the output tensor and returns a modified tensor
        """
        if hook_point not in self._hook_points:
            raise ValueError(f"Hook point '{hook_point}' not found")
        
        self._interventions[hook_point] = intervention_fn
    
    def remove_intervention(self, hook_point: str):
        """
        Remove an intervention function
        
        Args:
            hook_point: Name of the hook point
        """
        if hook_point in self._interventions:
            del self._interventions[hook_point]
    
    def run_with_cache(self, input_ids, attention_mask=None, 
                      names_filter: Optional[Union[str, List[str]]] = None,
                      **kwargs):
        """
        Run the model and cache activations
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            names_filter: Filter for which hook points to cache
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Tuple of (model outputs, cache)
        """
        # Set up hooks
        self._setup_hooks(names_filter)
        
        # Run the model
        if attention_mask is None and hasattr(input_ids, "shape") and len(input_ids.shape) > 1:
            # Create an attention mask if none is provided
            attention_mask = torch.ones_like(input_ids)
            
        with torch.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Get the cache
        cache = dict(self._cache)
        
        # Remove hooks
        self.remove_hooks()
        
        return outputs, cache
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """
        Call the model without caching activations
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Model outputs
        """
        return self._model(input_ids, attention_mask=attention_mask, **kwargs) 
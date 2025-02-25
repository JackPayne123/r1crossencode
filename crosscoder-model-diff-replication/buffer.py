#from utils import *
import tqdm
import time
import einops

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both models that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    Modified to work with HuggingFace models directly instead of TransformerLens.
    """

    def __init__(self, cfg, model_A, model_B, tokenized_data):
        """Initialize the buffer with the given configuration and models."""
        try:
            self.cfg = cfg
            self.model_A = model_A
            self.model_B = model_B
            
            # Ensure models have same hidden size
            if model_A.cfg.hidden_size != model_B.cfg.hidden_size:
                raise ValueError("Models must have the same hidden size")
                
            # Store hidden size in cfg if not present
            if "hidden_size" not in cfg:
                cfg["hidden_size"] = model_A.cfg.hidden_size
            
            # Store tokenized data references
            self.tokens_A = tokenized_data["tokens_A"]
            self.tokens_B = tokenized_data["tokens_B"]
            
            # Validate tokenized data
            if len(self.tokens_A) != len(self.tokens_B):
                raise ValueError(f"Tokenized data lengths don't match: A={len(self.tokens_A)}, B={len(self.tokens_B)}")
            
            print("\nValidating tokenized data:")
            print(f"Model A tokens shape: {self.tokens_A.shape}")
            print(f"Model B tokens shape: {self.tokens_B.shape}")
            print(f"Model A unique tokens: {self.tokens_A.unique().shape[0]}")
            print(f"Model B unique tokens: {self.tokens_B.unique().shape[0]}")
            
            # Get special tokens
            self.special_tokens_A = self._get_special_tokens(model_A)
            self.special_tokens_B = self._get_special_tokens(model_B)
            
            print("\nSpecial tokens:")
            print(f"Model A: {self.special_tokens_A}")
            print(f"Model B: {self.special_tokens_B}")
            
            # Calculate buffer size based on available data
            self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
            self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
            self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
            
            # Initialize buffer in very small chunks to manage memory
            chunk_size = 1000  # Much smaller chunk size
            print(f"Initializing buffer with size: {self.buffer_size} in chunks of {chunk_size}")
            
            try:
                # Initialize buffer on CPU first
                self.buffer = torch.zeros(
                    (self.buffer_size, 2, cfg["hidden_size"]),
                    dtype=torch.bfloat16,
                    device='cpu'  # Initialize on CPU
                )
                
                # Move to GPU in chunks
                print("Moving buffer to GPU in chunks...")
                for i in tqdm.trange(0, self.buffer_size, chunk_size):
                    end_idx = min(i + chunk_size, self.buffer_size)
                    # Move chunk to GPU
                    self.buffer[i:end_idx] = self.buffer[i:end_idx].to(cfg["device"])
                    # Optional: Sleep to allow memory to settle
                    if i % (chunk_size * 10) == 0:
                        torch.cuda.empty_cache()
                        time.sleep(0.01)
                
            except Exception as e:
                print(f"Error during buffer initialization: {str(e)}")
                raise
            
            self.pointer = 0
            self.token_pointer = 0
            self.first = True
            self.normalize = True  # Enable normalization by default
            self._norm_factor_device = False  # Track if norm factor has been moved to device
            
            # Initialize normalization factors
            print("Estimating norm scaling factors...")
            norm_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A, self.tokens_A)
            norm_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B, self.tokens_B)
            print(f"Normalization factors - Model A: {norm_A:.6f}, Model B: {norm_B:.6f}")
            
            self.normalisation_factor = torch.tensor(
                [norm_A, norm_B],
                device=cfg["device"],  # Initialize directly on device
                dtype=torch.float32,
            )
            self._norm_factor_device = True
            
            if torch.any(torch.isnan(self.normalisation_factor)) or torch.any(torch.isinf(self.normalisation_factor)):
                raise ValueError(f"Invalid normalization factors: {self.normalisation_factor}")
            
            print("Buffer initialized successfully!")
            
            # Verify hook point exists in both models
            print(f"Verifying hook point {self.cfg['hook_point']} exists in both models...")
            try:
                module_A = self.model_A._find_module(self.cfg["hook_point"])
                module_B = self.model_B._find_module(self.cfg["hook_point"])
                print(f"Hook point found in both models. Type A: {type(module_A)}, Type B: {type(module_B)}")
            except Exception as e:
                raise ValueError(f"Hook point {self.cfg['hook_point']} not found in models: {str(e)}")
            
        except Exception as e:
            print(f"Error in Buffer initialization: {str(e)}")
            raise

    def _get_special_tokens(self, model):
        """Get special tokens from a HuggingFace model's tokenizer."""
        try:
            special_tokens = {
                "bos": model.tokenizer.bos_token_id,
                "eos": model.tokenizer.eos_token_id,
                "pad": model.tokenizer.pad_token_id,
                "sep": model.tokenizer.sep_token_id if hasattr(model.tokenizer, 'sep_token_id') else None,
                "cls": model.tokenizer.cls_token_id if hasattr(model.tokenizer, 'cls_token_id') else None,
                "mask": model.tokenizer.mask_token_id if hasattr(model.tokenizer, 'mask_token_id') else None,
            }
            
            # Filter out None values for tokens that don't exist
            return {k: v for k, v in special_tokens.items() if v is not None}
            
        except Exception as e:
            print(f"Error getting special tokens: {str(e)}")
            # Return minimal set of special tokens if there's an error
            return {
                "bos": model.tokenizer.bos_token_id if hasattr(model.tokenizer, 'bos_token_id') else None,
                "eos": model.tokenizer.eos_token_id if hasattr(model.tokenizer, 'eos_token_id') else None,
                "pad": model.tokenizer.pad_token_id if hasattr(model.tokenizer, 'pad_token_id') else None
            }

    def _get_content_mask(self, tokens, special_tokens):
        """Create a mask for content tokens (excluding special tokens)"""
        mask = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in special_tokens.values():
            if token_id is not None:
                mask &= (tokens != token_id)
        return mask

    def _process_activations(self, tokens, cache, model, special_tokens, hook_point):
        """Process activations by removing special tokens and aligning content"""
        try:
            acts = cache[hook_point]
            print(f"Raw activations shape: {acts.shape}, mean: {acts.mean().item():.6f}, std: {acts.std().item():.6f}")
            print(f"Raw activations range: [{acts.min().item():.6f}, {acts.max().item():.6f}]")
            
            # Only drop BOS token like in original
            if len(acts.shape) == 3:  # [batch, seq_len, hidden]
                acts = acts[:, 1:, :]
            elif len(acts.shape) == 4:  # [n_layer, batch, seq_len, hidden]
                acts = acts[:, :, 1:, :]
                
            print(f"After BOS removal - shape: {acts.shape}, mean: {acts.mean().item():.6f}, std: {acts.std().item():.6f}")
            print(f"Range: [{acts.min().item():.6f}, {acts.max().item():.6f}]")
            
            return acts
            
        except Exception as e:
            print(f"Error in activation processing: {str(e)}")
            raise

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, tokens, n_batches_for_norm_estimate: int = 10):
        """Estimate normalization scaling factor for model activations."""
        norms_per_batch = []
        
        # Reduce batch size for norm estimation
        small_batch_size = 2  # Even smaller batch size for quantized models
        
        # Set up proper dtype for quantized models
        compute_dtype = torch.bfloat16  # Same as model's compute dtype from BitsAndBytesConfig
        
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            try:
                # Get a small batch
                batch_start = i * small_batch_size
                batch_end = min((i + 1) * small_batch_size, len(tokens))
                batch_tokens = tokens[batch_start:batch_end].to(model.device)
                
                if batch_tokens.shape[0] == 0:
                    continue
                    
                # Ensure consistent sequence length
                if batch_tokens.shape[1] > self.cfg["seq_len"]:
                    batch_tokens = batch_tokens[:, :self.cfg["seq_len"]]
                
                # Run with cache but immediately extract what we need
                # Keep everything in bfloat16 to match quantized model's compute dtype
                with torch.amp.autocast('cuda', dtype=compute_dtype):
                    _, cache = model.run_with_cache(
                        batch_tokens,
                        names_filter=self.cfg["hook_point"],
                    )
                    acts = cache[self.cfg["hook_point"]]
                    
                    # Drop BOS token and calculate norm immediately
                    if len(acts.shape) == 3:  # [batch, seq_len, hidden]
                        acts = acts[:, 1:, :]
                    elif len(acts.shape) == 4:  # [n_layer, batch, seq_len, hidden]
                        acts = acts[:, :, 1:, :]
                    
                    # Calculate norm and immediately detach and move to CPU
                    # Keep in bfloat16 until the last possible moment
                    batch_norms = acts.norm(dim=-1).mean().cpu().item()
                    norms_per_batch.append(batch_norms)
                
                # Clear cache
                del cache
                del acts
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue
            
            # Add a small sleep to allow GPU memory to properly clear
            time.sleep(0.01)
        
        if not norms_per_batch:
            raise ValueError("Failed to compute any norms")
            
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.hidden_size) / mean_norm
        return scaling_factor

    def _shuffle_buffer_in_chunks(self, chunk_size=1000):
        """Memory efficient in-place buffer shuffling."""
        buffer_size = self.buffer.shape[0]
        device = self.buffer.device
        
        try:
            print(f"Shuffling buffer of size {buffer_size} in chunks of {chunk_size}")
            
            # Generate permutation on CPU
            perm = torch.randperm(buffer_size, device='cpu')
            
            # Process in chunks
            for i in tqdm.trange(0, buffer_size, chunk_size):
                end_idx = min(i + chunk_size, buffer_size)
                chunk_size_actual = end_idx - i
                
                # Get source indices for this chunk
                chunk_perm = perm[i:end_idx].to(device)
                
                # Store original chunk
                temp = self.buffer[chunk_perm].clone()
                self.buffer[i:end_idx] = temp
                
                del temp, chunk_perm
                torch.cuda.empty_cache()
            
            print("Buffer shuffling complete")
            
        except Exception as e:
            print(f"Error during buffer shuffling: {str(e)}")
            raise

    @torch.no_grad()
    def refresh(self):
        """Refresh the buffer with new activations."""
        self.pointer = 0
        print("Refreshing the buffer!")
        
        with torch.autocast("cuda", torch.bfloat16):
            # Calculate number of batches needed
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            
            # Use model_batch_size from config
            batch_size = self.cfg["model_batch_size"]
            print(f"\nProcessing {num_batches} batches with batch size {batch_size}")
            
            for _ in tqdm.trange(0, num_batches, batch_size):
                try:
                    # Get batch of tokens for both models
                    end_idx = min(self.token_pointer + batch_size, len(self.tokens_A))
                    tokens_A = self.tokens_A[self.token_pointer:end_idx].to(self.cfg["device"])
                    tokens_B = self.tokens_B[self.token_pointer:end_idx].to(self.cfg["device"])
                    
                    # Skip if empty batch
                    if tokens_A.shape[0] == 0:
                        continue
                    
                    # Ensure consistent sequence length
                    if tokens_A.shape[1] > self.cfg["seq_len"]:
                        tokens_A = tokens_A[:, :self.cfg["seq_len"]]
                    if tokens_B.shape[1] > self.cfg["seq_len"]:
                        tokens_B = tokens_B[:, :self.cfg["seq_len"]]
                    
                    # Get activations from both models
                    _, cache_A = self.model_A.run_with_cache(
                        tokens_A,
                        names_filter=self.cfg["hook_point"],
                    )
                    
                    _, cache_B = self.model_B.run_with_cache(
                        tokens_B,
                        names_filter=self.cfg["hook_point"],
                    )
                    
                    # Extract activations and drop BOS token
                    acts_A = cache_A[self.cfg["hook_point"]][:, 1:, :]
                    acts_B = cache_B[self.cfg["hook_point"]][:, 1:, :]
                    
                    # Print stats periodically
                    if self.pointer % (batch_size * 10) == 0:
                        print(f"\nActivation stats at position {self.pointer}:")
                        print(f"Model A - mean: {acts_A.mean().item():.6f}, std: {acts_A.std().item():.6f}")
                        print(f"Model B - mean: {acts_B.mean().item():.6f}, std: {acts_B.std().item():.6f}")
                    
                    # Stack and reshape
                    acts = torch.stack([acts_A, acts_B], dim=0)
                    acts = einops.rearrange(
                        acts,
                        "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                    )
                    
                    # Update buffer
                    available_space = self.buffer.size(0) - self.pointer
                    if acts.shape[0] > 0:
                        if acts.shape[0] > available_space:
                            acts = acts[:available_space]
                        self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                        self.pointer += acts.shape[0]
                        self.token_pointer = end_idx
                    
                    # Clean up
                    del cache_A, cache_B, acts_A, acts_B, acts
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM error, reducing batch size")
                        batch_size = max(1, batch_size // 2)
                        print(f"New batch size: {batch_size}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"Error in refresh batch: {str(e)}")
                    continue
            
            # Reset token pointer if needed
            if self.token_pointer >= len(self.tokens_A) - batch_size:
                self.token_pointer = 0
            
            print("\nShuffling buffer...")
            self._shuffle_buffer_in_chunks(chunk_size=1000)
            
            # Print final stats
            print(f"\nFinal buffer statistics:")
            print(f"Mean: {self.buffer.mean().item():.6f}")
            print(f"Std: {self.buffer.std().item():.6f}")
            print(f"Range: [{self.buffer.min().item():.6f}, {self.buffer.max().item():.6f}]")
            
            self.pointer = 0
            print("Buffer refresh complete!")

    @torch.no_grad()
    def next(self):
        """Get next batch of activations from the buffer."""
        try:
            # Get data and verify it's non-zero
            start_idx = self.pointer
            end_idx = self.pointer + self.cfg["batch_size"]
            
            # Get slice and verify
            out = self.buffer[start_idx:end_idx]
            if out.abs().mean() < 1e-6:
                print(f"\nWarning: Retrieved near-zero batch from buffer at position {start_idx}")
                print("Attempting to find non-zero batch...")
                
                # Try to find a non-zero batch
                for test_idx in range(0, self.buffer.size(0) - self.cfg["batch_size"], self.cfg["batch_size"]):
                    test_batch = self.buffer[test_idx:test_idx + self.cfg["batch_size"]]
                    if test_batch.abs().mean() >= 1e-6:
                        print(f"Found non-zero batch at position {test_idx}")
                        out = test_batch
                        self.pointer = test_idx  # Update pointer to this position
                        break
            
            # Move to device and convert to float32 for training
            out = out.to(self.cfg["device"]).float()
            
            # Print pre-normalization stats
            print(f"\nPre-normalization batch stats:")
            print(f"Shape: {out.shape}")
            print(f"Mean: {out.mean().item():.6f}")
            print(f"Std: {out.std().item():.6f}")
            print(f"Range: [{out.min().item():.6f}, {out.max().item():.6f}]")
            print(f"Mean abs value: {out.abs().mean().item():.6f}")
            
            # Apply normalization if enabled
            if self.normalize:
                # Ensure normalization factor is on the correct device
                if not self._norm_factor_device:
                    self.normalisation_factor = self.normalisation_factor.to(self.cfg["device"])
                    self._norm_factor_device = True
                
                # Apply normalization with explicit broadcasting
                norm_factor = self.normalisation_factor[None, :, None]  # [1, 2, 1]
                out = out * norm_factor  # [batch, 2, d_model]
                
                # Print post-normalization stats
                print(f"\nPost-normalization batch stats:")
                print(f"Mean: {out.mean().item():.6f}")
                print(f"Std: {out.std().item():.6f}")
                print(f"Range: [{out.min().item():.6f}, {out.max().item():.6f}]")
                print(f"Mean abs value: {out.abs().mean().item():.6f}")
                print(f"Normalization factors: {self.normalisation_factor.tolist()}")
            
            # Update pointer and refresh if needed
            self.pointer += self.cfg["batch_size"]
            if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
                print("\nBuffer halfway depleted, refreshing...")
                self.refresh()
            
            # Final verification
            if out.isnan().any() or out.isinf().any():
                print("\nWarning: Found NaN or Inf values in output batch")
                print(f"NaN count: {out.isnan().sum().item()}")
                print(f"Inf count: {out.isinf().sum().item()}")
            
            return out
            
        except Exception as e:
            print(f"Error in next batch: {str(e)}")
            raise

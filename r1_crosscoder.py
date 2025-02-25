import torch
from torch import nn
import torch.nn.functional as F
import einops
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, NamedTuple
from huggingface_hub import hf_hub_download
import pprint

# Define supported data types
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

class LossOutput(NamedTuple):
    """Container for various loss metrics from the crosscoder."""
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoder(nn.Module):
    """
    Implementation of a crosscoder model for analyzing differences between two models.
    Adapted for R1 and Gemma model comparison with improved error handling and monitoring.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the crosscoder model.
        
        Args:
            cfg: Configuration dictionary containing model parameters
        """
        try:
            super().__init__()
            self.cfg = cfg
            d_hidden = self.cfg["dict_size"]
            d_in = self.cfg["d_in"]
            self.dtype = DTYPES[self.cfg["enc_dtype"]]
            
            # Set random seed for reproducibility
            torch.manual_seed(self.cfg["seed"])
            
            # Initialize encoder weights (2 models)
            self.W_enc = nn.Parameter(
                torch.empty(2, d_in, d_hidden, dtype=self.dtype)
            )
            
            # Initialize decoder weights with normal distribution
            self.W_dec = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(d_hidden, 2, d_in, dtype=self.dtype)
                )
            )
            
            # Normalize decoder weights
            self.W_dec.data = (
                self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
            )
            
            # Initialize encoder weights as transpose of decoder
            self.W_enc.data = einops.rearrange(
                self.W_dec.data.clone(),
                "d_hidden n_models d_model -> n_models d_model d_hidden",
            )
            
            # Initialize biases
            self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
            self.b_dec = nn.Parameter(torch.zeros((2, d_in), dtype=self.dtype))
            
            self.d_hidden = d_hidden
            
            # Move model to specified device
            self.to(self.cfg["device"])
            
            # Initialize saving attributes
            self.save_dir = None
            self.save_version = 0
            
            print(f"\nInitialized CrossCoder with configuration:")
            print(f"Hidden size: {d_hidden}")
            print(f"Input size: {d_in}")
            print(f"Device: {self.cfg['device']}")
            print(f"Dtype: {self.dtype}")
            
        except Exception as e:
            print(f"Error initializing CrossCoder: {str(e)}")
            raise

    def encode(self, x: torch.Tensor, apply_relu: bool = True) -> torch.Tensor:
        """
        Encode input activations through the encoder network.
        
        Args:
            x: Input tensor of shape [batch, n_models, d_model]
            apply_relu: Whether to apply ReLU activation
            
        Returns:
            Encoded activations
        """
        try:
            x_enc = einops.einsum(
                x,
                self.W_enc,
                "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
            )
            
            if apply_relu:
                acts = F.relu(x_enc + self.b_enc)
            else:
                acts = x_enc + self.b_enc
                
            return acts
            
        except Exception as e:
            print(f"Error in encode: {str(e)}")
            raise

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Decode latent activations back to model space.
        
        Args:
            acts: Encoded activations of shape [batch, d_hidden]
            
        Returns:
            Decoded activations
        """
        try:
            acts_dec = einops.einsum(
                acts,
                self.W_dec,
                "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
            )
            return acts_dec + self.b_dec
            
        except Exception as e:
            print(f"Error in decode: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the crosscoder."""
        return self.decode(self.encode(x))

    def get_losses(self, x: torch.Tensor) -> LossOutput:
        """
        Calculate various loss metrics for the crosscoder.
        
        Args:
            x: Input tensor of shape [batch, n_models, d_model]
            
        Returns:
            LossOutput containing various loss metrics
        """
        try:
            # Print input statistics
            print(f"\nInput shape: {x.shape}")
            print(f"Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
            
            # Convert to correct dtype
            x = x.to(self.dtype)
            
            # Get encoded activations
            acts = self.encode(x)
            print(f"Encoded shape: {acts.shape}")
            print(f"Encoded mean: {acts.mean().item():.6f}, std: {acts.std().item():.6f}")
            
            # Reconstruct input
            x_reconstruct = self.decode(acts)
            print(f"Reconstructed shape: {x_reconstruct.shape}")
            print(f"Reconstructed mean: {x_reconstruct.mean().item():.6f}, std: {x_reconstruct.std().item():.6f}")
            
            # Calculate reconstruction error
            diff = x_reconstruct.float() - x.float()
            squared_diff = diff.pow(2)
            l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
            l2_loss = l2_per_batch.mean()
            
            # Calculate explained variance
            total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
            explained_variance = 1 - l2_per_batch / total_variance
            
            # Calculate per-model metrics
            per_token_l2_loss_A = (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
            total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
            explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
            
            per_token_l2_loss_B = (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
            total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
            explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B
            
            # Calculate L1 loss using decoder norms
            decoder_norms = self.W_dec.norm(dim=-1)
            total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
            l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
            
            # Calculate L0 loss (sparsity)
            l0_loss = (acts > 0).float().sum(-1).mean()
            
            # Print loss statistics
            print(f"\nLoss Statistics:")
            print(f"L2 Loss: {l2_loss.item():.6f}")
            print(f"L1 Loss: {l1_loss.item():.6f}")
            print(f"L0 Loss: {l0_loss.item():.6f}")
            print(f"Explained Variance: {explained_variance.mean().item():.6f}")
            print(f"Explained Variance A: {explained_variance_A.mean().item():.6f}")
            print(f"Explained Variance B: {explained_variance_B.mean().item():.6f}")
            
            return LossOutput(
                l2_loss=l2_loss,
                l1_loss=l1_loss,
                l0_loss=l0_loss,
                explained_variance=explained_variance,
                explained_variance_A=explained_variance_A,
                explained_variance_B=explained_variance_B
            )
            
        except Exception as e:
            print(f"Error calculating losses: {str(e)}")
            raise

    def create_save_dir(self) -> None:
        """Create a versioned directory for saving checkpoints."""
        try:
            # Create base checkpoint directory
            base_dir = Path("./checkpoints")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Get existing versions
            version_list = [
                int(file.name.split("_")[1])
                for file in base_dir.iterdir()
                if "version" in str(file)
            ]
            
            # Create new version
            version = 1 + max(version_list) if version_list else 0
            self.save_dir = base_dir / f"version_{version}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Created checkpoint directory at {self.save_dir}")
            
        except Exception as e:
            print(f"Error creating save directory: {str(e)}")
            # Fallback to simple directory
            self.save_dir = Path("./checkpoints/version_0")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created fallback checkpoint directory at {self.save_dir}")

    def save(self) -> None:
        """Save model checkpoint and configuration."""
        try:
            if self.save_dir is None:
                self.create_save_dir()
                
            # Save paths
            weight_path = self.save_dir / f"{self.save_version}.pt"
            cfg_path = self.save_dir / f"{self.save_version}_cfg.json"
            
            # Save model weights
            torch.save(self.state_dict(), str(weight_path))
            
            # Save configuration
            with open(str(cfg_path), "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
                
            print(f"Saved checkpoint {self.save_version} to {self.save_dir}")
            self.save_version += 1
            
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            # Try fallback save
            try:
                fallback_path = Path(f"crosscoder_checkpoint_{self.save_version}.pt")
                torch.save(self.state_dict(), str(fallback_path))
                print(f"Saved fallback checkpoint to {fallback_path}")
                self.save_version += 1
            except Exception as e2:
                print(f"Failed to save fallback checkpoint: {str(e2)}")

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            device: Device to load the model to
            
        Returns:
            Initialized CrossCoder instance
        """
        try:
            # Download files from HuggingFace
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{path}/cfg.json"
            )
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{path}/cc_weights.pt"
            )
            
            # Load configuration
            with open(config_path, 'r', encoding="utf-8") as f:
                cfg = json.load(f)
            
            # Override device if specified
            if device is not None:
                cfg["device"] = str(device)
            
            # Initialize model
            print("\nInitializing CrossCoder from HuggingFace checkpoint")
            print("Configuration:")
            pprint.pprint(cfg)
            
            instance = cls(cfg)
            
            # Load weights
            state_dict = torch.load(weights_path, map_location=cfg["device"])
            instance.load_state_dict(state_dict)
            
            print(f"Successfully loaded CrossCoder from {repo_id}")
            return instance
            
        except Exception as e:
            print(f"Error loading from HuggingFace: {str(e)}")
            raise

    @classmethod
    def load(cls, version_dir: str, checkpoint_version: int) -> "CrossCoder":
        """
        Load a local checkpoint.
        
        Args:
            version_dir: Directory containing the version
            checkpoint_version: Specific checkpoint version to load
            
        Returns:
            Initialized CrossCoder instance
        """
        try:
            # Construct paths
            load_dir = Path("./checkpoints") / str(version_dir)
            cfg_path = load_dir / f"{str(checkpoint_version)}_cfg.json"
            weight_path = load_dir / f"{str(checkpoint_version)}.pt"
            
            # Load configuration
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            
            print("\nLoading local checkpoint")
            print("Configuration:")
            pprint.pprint(cfg)
            
            # Initialize model and load weights
            instance = cls(cfg)
            instance.load_state_dict(torch.load(weight_path))
            
            print(f"Successfully loaded checkpoint from {load_dir}")
            return instance
            
        except Exception as e:
            print(f"Error loading local checkpoint: {str(e)}")
            raise 
from utils import *

from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download
from typing import NamedTuple
from google.colab import drive
import os

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/crosscoder-model-diff-replication/checkpoints")

class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        # hardcoding n_models to 2
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((2, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0
        
        # Try to mount Google Drive if not already mounted
        try:
            drive.mount('/content/drive', force_remount=False)
            print("Google Drive already mounted")
        except:
            print("Mounting Google Drive...")
            drive.mount('/content/drive')

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
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

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        print(f"\nInput shape: {x.shape}, mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
        x = x.to(self.dtype)
        acts = self.encode(x)
        print(f"Encoded acts shape: {acts.shape}, mean: {acts.mean().item():.6f}, std: {acts.std().item():.6f}")
        
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        print(f"Reconstructed shape: {x_reconstruct.shape}, mean: {x_reconstruct.mean().item():.6f}, std: {x_reconstruct.std().item():.6f}")
        
        diff = x_reconstruct.float() - x.float()
        print(f"Diff mean: {diff.mean().item():.6f}, std: {diff.std().item():.6f}")
        
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()
        print(f"L2 loss: {l2_loss.item():.6f}")

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance
        print(f"Explained variance mean: {explained_variance.mean().item():.6f}")

        per_token_l2_loss_A = (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A
        print(f"Explained variance A mean: {explained_variance_A.mean().item():.6f}")

        per_token_l2_loss_B = (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B
        print(f"Explained variance B mean: {explained_variance_B.mean().item():.6f}")

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        print(f"L1 loss: {l1_loss.item():.6f}")

        l0_loss = (acts>0).float().sum(-1).mean()
        print(f"L0 loss: {l0_loss.item():.6f}")

        return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss, explained_variance=explained_variance, explained_variance_A=explained_variance_A, explained_variance_B=explained_variance_B)

    def create_save_dir(self):
        """Create a versioned directory for saving checkpoints in Google Drive"""
        try:
            # Use Google Drive path
            base_dir = Path("/content/drive/MyDrive/crosscoder_checkpoints")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Get list of existing versions
            version_list = [
                int(file.name.split("_")[1])
                for file in list(base_dir.iterdir())
                if "version" in str(file)
            ]
            
            # Create new version
            if len(version_list):
                version = 1 + max(version_list)
            else:
                version = 0
            
            self.save_dir = base_dir / f"version_{version}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created checkpoint directory at {self.save_dir}")
            
        except Exception as e:
            print(f"Error creating save directory in Drive: {str(e)}")
            # Fallback to a simple directory structure in Drive
            self.save_dir = Path("/content/drive/MyDrive/crosscoder_checkpoints/version_0")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created fallback checkpoint directory at {self.save_dir}")

    def save(self):
        """Save model checkpoint to Google Drive"""
        try:
            if self.save_dir is None:
                self.create_save_dir()
            
            weight_path = self.save_dir / f"{self.save_version}.pt"
            cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

            # Save model weights
            print(f"\nSaving checkpoint {self.save_version} to Drive...")
            torch.save(self.state_dict(), str(weight_path))
            
            # Save config
            with open(str(cfg_path), "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)

            print(f"Successfully saved checkpoint to {self.save_dir}")
            self.save_version += 1
            
        except Exception as e:
            print(f"Error saving checkpoint to Drive: {str(e)}")
            # Try to save in local colab directory as fallback
            try:
                local_path = Path("/content/crosscoder_backup")
                local_path.mkdir(exist_ok=True)
                backup_path = local_path / f"crosscoder_checkpoint_{self.save_version}.pt"
                torch.save(self.state_dict(), backup_path)
                print(f"Saved fallback checkpoint locally at {backup_path}")
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
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)
            
        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance

    @classmethod
    def load(cls, version_dir, checkpoint_version):
        """Load model checkpoint from Google Drive or local backup"""
        # Try Google Drive first
        drive_path = Path("/content/drive/MyDrive/crosscoder_checkpoints") / str(version_dir)
        local_path = Path("/content/crosscoder_backup")
        
        try:
            # Try Drive first
            if drive_path.exists():
                cfg_path = drive_path / f"{str(checkpoint_version)}_cfg.json"
                weight_path = drive_path / f"{str(checkpoint_version)}.pt"
                print(f"Loading checkpoint from Drive: {weight_path}")
            # Fall back to local backup
            elif local_path.exists():
                cfg_path = local_path / f"{str(checkpoint_version)}_cfg.json"
                weight_path = local_path / f"crosscoder_checkpoint_{checkpoint_version}.pt"
                print(f"Loading checkpoint from local backup: {weight_path}")
            else:
                raise FileNotFoundError(f"Could not find checkpoint in Drive or local backup")

            # Load config
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            
            # Initialize model
            print("\nModel configuration:")
            pprint.pprint(cfg)
            self = cls(cfg=cfg)
            
            # Load weights
            print("\nLoading model weights...")
            self.load_state_dict(torch.load(weight_path))
            print("Successfully loaded model checkpoint")
            
            return self
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise
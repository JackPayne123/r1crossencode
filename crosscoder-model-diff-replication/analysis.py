# %%
from utils import *
from crosscoder import CrossCoder
import plotly.express as px
import plotly.io as pio
import torch
import tqdm
import torch.nn.functional as F

# Configure plotly to work in Colab
pio.renderers.default = "colab"
torch.set_grad_enabled(False)

# %%
# Load locally trained crosscoder
# Replace version_dir and checkpoint_version with your actual values
VERSION_DIR = "version_0"  # The version directory containing your trained model
CHECKPOINT_VERSION = 0     # The checkpoint number you want to load

try:
    cross_coder = CrossCoder.load(VERSION_DIR, CHECKPOINT_VERSION)
    print(f"Successfully loaded crosscoder from version {VERSION_DIR}, checkpoint {CHECKPOINT_VERSION}")
except Exception as e:
    print(f"Error loading local crosscoder: {str(e)}")
    print("Falling back to HuggingFace model")
    cross_coder = CrossCoder.load_from_hf()

# %%
# Calculate norms and print shape for debugging
norms = cross_coder.W_dec.norm(dim=-1)
print("Norms shape:", norms.shape)

# %%
# Calculate relative norms and print shape for debugging
relative_norms = norms[:, 1] / norms.sum(dim=-1)
print("Relative norms shape:", relative_norms.shape)

# %%
# First histogram - Relative decoder norm strength
fig1 = px.histogram(
    relative_norms.detach().cpu().numpy(), 
    title="Gemma 2 2B Base vs IT Model Diff",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig1.update_layout(
    showlegend=False,
    title_x=0.5,  # Center the title
    yaxis_title="Number of Latents",
    xaxis=dict(
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=['0', '0.25', '0.5', '0.75', '1.0']
    )
)

# Display the figure
fig1.show()

# %%
# Calculate shared latent mask and print shape for debugging
shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)
print("Shared latent mask shape:", shared_latent_mask.shape)
print("Number of shared latents:", shared_latent_mask.sum().item())

# %%
# Calculate cosine similarities and print shape for debugging
cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (
    cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1)
)
print("Cosine similarities shape:", cosine_sims.shape)

# %%
# Second histogram - Cosine similarity
fig2 = px.histogram(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(), 
    title="Cosine Similarity Distribution of Shared Features",
    labels={"value": "Cosine similarity of decoder vectors between models"},
    log_y=True,
    range_x=[-1, 1],
    nbins=100,
)

fig2.update_layout(
    showlegend=False,
    title_x=0.5,  # Center the title
    yaxis_title="Number of Latents (log scale)",
    xaxis_title="Cosine Similarity"
)

# Display the figure
fig2.show()

# Print summary statistics
print("\nSummary Statistics:")
cos_sims_filtered = cosine_sims[shared_latent_mask]
print(f"Mean cosine similarity: {cos_sims_filtered.mean().item():.3f}")
print(f"Median cosine similarity: {cos_sims_filtered.median().item():.3f}")
print(f"Std cosine similarity: {cos_sims_filtered.std().item():.3f}")
# %%

def calculate_ce_metrics(crosscoder, model_A, model_B, tokenized_data, num_sequences=40, seq_length=1024, batch_size=2):
    """Calculate CE recovery metrics as defined in the paper, processing in small batches"""
    try:
        # Sample random sequences
        total_seqs = len(tokenized_data['tokens_A'])
        indices = torch.randperm(total_seqs)[:num_sequences]
        
        # Initialize accumulators
        ce_recon_A_total = 0
        ce_recon_B_total = 0
        ce_id_A_total = 0
        ce_id_B_total = 0
        ce_zero_A_total = 0
        ce_zero_B_total = 0
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        print(f"\nProcessing {num_sequences} sequences in {num_batches} batches of size {batch_size}...")
        
        for i in tqdm.trange(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)
            batch_indices = indices[i:batch_end]
            
            # Get batch of tokens
            tokens_A = tokenized_data['tokens_A'][batch_indices][:, :seq_length].to(crosscoder.cfg["device"])
            tokens_B = tokenized_data['tokens_B'][batch_indices][:, :seq_length].to(crosscoder.cfg["device"])
            
            with torch.no_grad():
                # Get original activations
                _, cache_A = model_A.run_with_cache(tokens_A, names_filter=crosscoder.cfg["hook_point"])
                _, cache_B = model_B.run_with_cache(tokens_B, names_filter=crosscoder.cfg["hook_point"])
                
                # Convert activations to float32
                acts_A = cache_A[crosscoder.cfg["hook_point"]].to(torch.float32)
                acts_B = cache_B[crosscoder.cfg["hook_point"]].to(torch.float32)
                
                # Print shapes and dtypes for debugging
                print(f"\nActivation shapes and dtypes:")
                print(f"Acts A: {acts_A.shape}, {acts_A.dtype}")
                print(f"Acts B: {acts_B.shape}, {acts_B.dtype}")
                
                # Handle layer dimension if present
                if len(acts_A.shape) == 4:  # [n_layer, batch, seq_len, hidden]
                    acts_A = acts_A[0]  # Take first layer if multiple
                    acts_B = acts_B[0]
                
                # Center the activations (apply pre-encoder bias)
                acts_A_center = acts_A - acts_A.mean(dim=-1, keepdim=True)
                acts_B_center = acts_B - acts_B.mean(dim=-1, keepdim=True)
                
                # Stack activations
                acts = torch.stack([acts_A_center, acts_B_center], dim=1)  # [batch, 2, seq_len, hidden]
                
                # Reshape to match crosscoder's expected input
                batch_size, n_models, seq_len, hidden = acts.shape
                acts = acts.reshape(batch_size * seq_len, n_models, hidden)
                
                # Get reconstructed activations through crosscoder
                # First get gating encoder (which features are active)
                active_features = crosscoder.encode(acts, apply_relu=False)  # Don't apply ReLU yet
                active_features = (active_features > 0).float()  # Binary gating
                
                # Get feature magnitudes
                feature_magnitudes = F.relu(crosscoder.encode(acts, apply_relu=False))
                
                # Combine gating and magnitudes
                reconstructed = crosscoder.decode(active_features * feature_magnitudes)
                
                # Calculate CE losses
                # Original CE loss (CE(Id))
                ce_id_A = -torch.sum(F.log_softmax(acts_A_center.reshape(-1, hidden), dim=-1) * 
                                   F.softmax(acts_A_center.reshape(-1, hidden), dim=-1), dim=-1).mean()
                ce_id_B = -torch.sum(F.log_softmax(acts_B_center.reshape(-1, hidden), dim=-1) * 
                                   F.softmax(acts_B_center.reshape(-1, hidden), dim=-1), dim=-1).mean()
                
                # Zero ablation CE loss (CE(ζ))
                zero_acts = torch.zeros_like(acts)
                ce_zero_A = -torch.sum(F.log_softmax(zero_acts[:, 0], dim=-1) * 
                                     F.softmax(acts[:, 0], dim=-1), dim=-1).mean()
                ce_zero_B = -torch.sum(F.log_softmax(zero_acts[:, 1], dim=-1) * 
                                     F.softmax(acts[:, 1], dim=-1), dim=-1).mean()
                
                # Reconstruction CE loss (CE(x ∘ f))
                ce_recon_A = -torch.sum(F.log_softmax(reconstructed[:, 0], dim=-1) * 
                                      F.softmax(acts[:, 0], dim=-1), dim=-1).mean()
                ce_recon_B = -torch.sum(F.log_softmax(reconstructed[:, 1], dim=-1) * 
                                      F.softmax(acts[:, 1], dim=-1), dim=-1).mean()
                
                # Accumulate values
                ce_recon_A_total += ce_recon_A.item() * (batch_end - i)
                ce_recon_B_total += ce_recon_B.item() * (batch_end - i)
                ce_id_A_total += ce_id_A.item() * (batch_end - i)
                ce_id_B_total += ce_id_B.item() * (batch_end - i)
                ce_zero_A_total += ce_zero_A.item() * (batch_end - i)
                ce_zero_B_total += ce_zero_B.item() * (batch_end - i)
                
                # Clear cache after each batch
                del cache_A, cache_B, acts_A, acts_B, acts, reconstructed, zero_acts
                torch.cuda.empty_cache()
        
        # Calculate final averages
        ce_recon_A = ce_recon_A_total / num_sequences
        ce_recon_B = ce_recon_B_total / num_sequences
        ce_id_A = ce_id_A_total / num_sequences
        ce_id_B = ce_id_B_total / num_sequences
        ce_zero_A = ce_zero_A_total / num_sequences
        ce_zero_B = ce_zero_B_total / num_sequences
        
        # Calculate CE recovery percentage using the paper's formula
        # loss_recovered = 1 - (CE(x ∘ f) - CE(Id)) / (CE(ζ) - CE(Id))
        base_ce_recovery = (1 - (ce_recon_A - ce_id_A) / (ce_zero_A - ce_id_A)) * 100
        chat_ce_recovery = (1 - (ce_recon_B - ce_id_B) / (ce_zero_B - ce_id_B)) * 100
        
        # Calculate CE delta
        base_ce_delta = ce_recon_A - ce_id_A
        chat_ce_delta = ce_recon_B - ce_id_B
        
        print("\nCE Loss Recovery Metrics:")
        print(f"Base Model CE Recovery: {base_ce_recovery:.2f}%")
        print(f"Chat Model CE Recovery: {chat_ce_recovery:.2f}%")
        print(f"Base Model CE Delta: {base_ce_delta:.3f}")
        print(f"Chat Model CE Delta: {chat_ce_delta:.3f}")
        
        # Print detailed metrics for debugging
        print("\nDetailed Metrics:")
        print(f"Base Model - CE(Id): {ce_id_A:.3f}, CE(ζ): {ce_zero_A:.3f}, CE(x ∘ f): {ce_recon_A:.3f}")
        print(f"Chat Model - CE(Id): {ce_id_B:.3f}, CE(ζ): {ce_zero_B:.3f}, CE(x ∘ f): {ce_recon_B:.3f}")
        
        return {
            "base_ce_recovery": base_ce_recovery,
            "chat_ce_recovery": chat_ce_recovery,
            "base_ce_delta": base_ce_delta,
            "chat_ce_delta": chat_ce_delta
        }
        
    except Exception as e:
        print(f"Error calculating CE metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Use in your analysis script:
print("\nCalculating CE metrics on 40 random sequences...")
ce_metrics = calculate_ce_metrics(
    crosscoder=cross_coder,  # Your loaded crosscoder
    model_A=base_model,      # Your base model
    model_B=cot_model,       # Your chat/CoT model
    tokenized_data=tokenized_data,  # Your tokenized dataset
    num_sequences=40,
    seq_length=1024,
    batch_size=2  # Process just 2 sequences at a time
)

# %%
# Get indices of latents in the rightmost bucket (0.97-1.005)
right_bucket_mask = (relative_norms >= 0.97) & (relative_norms <= 1.005)
right_bucket_indices = torch.where(right_bucket_mask)[0]

print(f"\nLatents with relative norm strength 0.97-1.005:")
print(f"Number of latents in range: {len(right_bucket_indices)}")
print(f"Indices: {right_bucket_indices.tolist()}")

# Print detailed information about these latents
print("\nDetailed information about these latents:")
for idx in right_bucket_indices:
    norm_A = norms[idx, 0].item()  # Base model norm
    norm_B = norms[idx, 1].item()  # IT model norm
    relative_norm = relative_norms[idx].item()
    print(f"Latent {idx}:")
    print(f"  Base model norm: {norm_A:.4f}")
    print(f"  IT model norm: {norm_B:.4f}")
    print(f"  Relative norm: {relative_norm:.4f}")

# Get a random batch of tokens
rand_idx = torch.randperm(len(tokenized_data['tokens_A']))[0]  # Get single random index
tokens_A = tokenized_data['tokens_A'][rand_idx:rand_idx+1]  # Get single sequence, keep batch dimension
tokens_B = tokenized_data['tokens_B'][rand_idx:rand_idx+1]  # Get corresponding sequence from model B

# Move tokens to the same device as the models and convert to float32
device = next(base_model._model.parameters()).device  # Get model's device
tokens_A = tokens_A.to(device)
tokens_B = tokens_B.to(device)

# Convert CrossCoder to float32 for compatibility
cross_coder = cross_coder.to(torch.float32)

print(f"\nToken shapes, devices and dtypes:")
print(f"Tokens A: {tokens_A.shape}, Device: {tokens_A.device}, Dtype: {tokens_A.dtype}")
print(f"Tokens B: {tokens_B.shape}, Device: {tokens_B.device}, Dtype: {tokens_B.dtype}")
print(f"Model device: {device}")
print(f"CrossCoder W_enc dtype: {cross_coder.W_enc.dtype}")
print(f"CrossCoder W_dec dtype: {cross_coder.W_dec.dtype}")

# Calculate metrics for both models
ce_metrics = get_ce_recovered_metrics(
    tokens=tokens_A,  # We'll use tokens_A as the input tokens
    model_A=base_model,
    model_B=cot_model,
    cross_coder=cross_coder
)

print("\nCE Recovery Metrics:")
for key, value in ce_metrics.items():
    print(f"{key}: {value}")

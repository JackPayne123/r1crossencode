# %%
from utils import *
from trainer import Trainer
import torch
from transformers import BitsAndBytesConfig

# %%
device = 'cuda:0'

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Initializing models...")

# Initialize our models using the ObservableModel wrapper
base_model = ObservableModel(
    "unsloth/Qwen2.5-14B-bnb-4bit", 
    device=device,
    trust_remote_code=True,
    quantization_config=quantization_config
)

cot_model = ObservableModel(
    "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit", 
    device=device,
    trust_remote_code=True,
    quantization_config=quantization_config
)

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 2_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.hidden_size,  # 5120 for Qwen models
    "dict_size": 2**14,
    "seq_len": 2048,
    "enc_dtype": "fp32",
    "model_name": "unsloth/Qwen2.5-14B-bnb-4bit",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "model.layers.24.input_layernorm",  # Middle layer for 48-layer model
    "wandb_project": "YOUR_WANDB_PROJECT",
    "wandb_entity": "YOUR_WANDB_ENTITY",
}

# Verify hook point exists
def verify_hook_point(model, hook_point):
    try:
        module = model._find_module(hook_point)
        print(f"Successfully found hook point: {hook_point}")
        print(f"Module type: {type(module)}")
        return True
    except Exception as e:
        print(f"Error finding hook point {hook_point}: {str(e)}")
        return False

if not verify_hook_point(base_model, default_cfg["hook_point"]):
    raise ValueError("Invalid hook point specified")

cfg = arg_parse_update_cfg(default_cfg)

# Load and pre-tokenize data for both models using the config
print("Loading and tokenizing data...")
tokenized_data = load_pile_lmsys_mixed_tokens(base_model, cot_model)

# Verify tokenized data
print("\nVerifying tokenized data:")
print(f"Tokens A shape: {tokenized_data['tokens_A'].shape}")
print(f"Tokens B shape: {tokenized_data['tokens_B'].shape}")
print(f"Tokens A range: [{tokenized_data['tokens_A'].min().item()}, {tokenized_data['tokens_A'].max().item()}]")
print(f"Tokens B range: [{tokenized_data['tokens_B'].min().item()}, {tokenized_data['tokens_B'].max().item()}]")

# Test model outputs
print("\nTesting model outputs:")
test_tokens_A = tokenized_data['tokens_A'][:2].to(device)
test_tokens_B = tokenized_data['tokens_B'][:2].to(device)

with torch.no_grad():
    print("\nModel A test:")
    _, cache_A = base_model.run_with_cache(
        test_tokens_A,
        names_filter=default_cfg["hook_point"]
    )
    print(f"Cache A keys: {list(cache_A.keys())}")
    acts_A = cache_A[default_cfg["hook_point"]]
    print(f"Acts A shape: {acts_A.shape}, mean: {acts_A.mean().item():.6f}, std: {acts_A.std().item():.6f}")
    
    print("\nModel B test:")
    _, cache_B = cot_model.run_with_cache(
        test_tokens_B,
        names_filter=default_cfg["hook_point"]
    )
    print(f"Cache B keys: {list(cache_B.keys())}")
    acts_B = cache_B[default_cfg["hook_point"]]
    print(f"Acts B shape: {acts_B.shape}, mean: {acts_B.mean().item():.6f}, std: {acts_B.std().item():.6f}")

print("\nStarting training...")
trainer = Trainer(cfg, base_model, cot_model, tokenized_data)
trainer.train()
# %%
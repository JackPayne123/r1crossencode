#from utils import *
#from crosscoder import CrossCoder
#from buffer import Buffer
import tqdm

from torch.nn.utils import clip_grad_norm_
class Trainer:
    def __init__(self, cfg, model_A, model_B, all_tokens):
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.all_tokens = all_tokens  # Store tokens but don't create buffer yet
        self.crosscoder = CrossCoder(cfg)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]
        self.buffer = None  # Initialize buffer as None
        self.global_step = 0  # Rename step_counter to global_step for clarity
        self.log_every = cfg.get("log_every", 100)  # Get log frequency from config or default to 100
        self.use_wandb = cfg.get("use_wandb", True)  # Enable wandb by default

        # Store model names for logging
        self.model_A_name = getattr(model_A, 'name_or_path', model_A._model.name_or_path)
        self.model_B_name = getattr(model_B, 'name_or_path', model_B._model.name_or_path)

        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

    def initialize_wandb(self):
        """Initialize wandb after buffer setup"""
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is None:
                    # Add model names to config
                    wandb_config = dict(self.cfg)
                    wandb_config.update({
                        "model_A": self.model_A_name,
                        "model_B": self.model_B_name,
                        "model_A_hidden_size": self.model_A.cfg.hidden_size,
                        "model_B_hidden_size": self.model_B.cfg.hidden_size,
                        "model_A_num_layers": self.model_A.cfg.num_hidden_layers,
                        "model_B_num_layers": self.model_B.cfg.num_hidden_layers,
                    })
                    
                    # Initialize wandb with enhanced config
                    wandb.init(
                        project=self.cfg.get("wandb_project", "crosscoder"),
                        entity=self.cfg.get("wandb_entity", None),
                        config=wandb_config,
                        name=self.cfg.get("run_name", None)
                    )
                    
                    # Log model comparison table with all values as strings
                    wandb.log({
                        "model_comparison": wandb.Table(
                            columns=["Property", "Model A", "Model B"],
                            data=[
                                ["Name", str(self.model_A_name), str(self.model_B_name)],
                                ["Hidden Size", str(self.model_A.cfg.hidden_size), str(self.model_B.cfg.hidden_size)],
                                ["Num Layers", str(self.model_A.cfg.num_hidden_layers), str(self.model_B.cfg.num_hidden_layers)],
                                ["Hook Point", str(self.cfg["hook_point"]), str(self.cfg["hook_point"])],
                            ]
                        )
                    })
                    
                print("Successfully initialized wandb with model information")
            except Exception as e:
                print(f"Failed to initialize wandb: {str(e)}")
                print("Continuing without wandb logging")
                self.use_wandb = False

    def initialize_buffer(self):
        """Initialize buffer only when needed"""
        if self.buffer is None:
            print("Initializing buffer...")
            try:
                self.buffer = Buffer(self.cfg, self.model_A, self.model_B, self.all_tokens)
                # Clear references to raw tokens after buffer is created
                self.all_tokens = None
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error initializing buffer: {str(e)}")
                raise

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.global_step < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.global_step / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        """Get next batch and compute loss"""
        acts = self.buffer.next()
        losses = self.crosscoder.get_losses(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        return loss_dict

    def log(self, loss_dict, force_print=False):
        """Log metrics to wandb and console."""
        if self.use_wandb:
            try:
                # Ensure all values are scalar for wandb
                wandb_metrics = {}
                for key, value in loss_dict.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[key] = value
                    elif isinstance(value, torch.Tensor):
                        wandb_metrics[key] = value.item()
                
                # Add step counter and model info
                wandb_metrics.update({
                    "global_step": self.global_step,
                    "epoch": self.global_step / self.total_steps,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "model_A": self.model_A_name,
                    "model_B": self.model_B_name,
                })
                
                # Log to wandb with explicit step and commit=True for immediate update
                import wandb
                wandb.log(wandb_metrics, step=self.global_step, commit=True)
                
                # Force wandb to sync the metrics immediately
                if hasattr(wandb.run, 'sync'):
                    wandb.run.sync()
                
                # Print to console periodically or if forced
                if force_print or self.global_step % self.log_every == 0:
                    print(f"\nStep {self.global_step}/{self.total_steps}")
                    print(f"Models: {self.model_A_name} vs {self.model_B_name}")
                    for key, value in wandb_metrics.items():
                        print(f"{key}: {value}")
            except Exception as e:
                print(f"Error during wandb logging: {str(e)}")
                print("Continuing without wandb logging for this step")
        else:
            # Always print if wandb is disabled
            print(f"\nStep {self.global_step}/{self.total_steps}")
            print(f"Models: {self.model_A_name} vs {self.model_B_name}")
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")

    def save(self):
        self.crosscoder.save()

    def train(self):
        """Initialize buffer at start of training and begin training loop"""
        print("\nStarting training...")
        print(f"Total steps: {self.total_steps}")
        print(f"Logging every {self.log_every} steps")
        
        # First initialize buffer (this will calculate norm factors)
        self.initialize_buffer()
        
        # Do a complete buffer refresh before starting training
        print("\nDoing initial buffer refresh...")
        self.buffer.refresh()
        
        # Then initialize wandb after buffer is ready
        self.initialize_wandb()
        
        # Reset step counter
        self.global_step = 0
        
        try:
            for i in tqdm.trange(self.total_steps):
                # Get loss dict from training step
                loss_dict = self.step()
                
                # Increment counter before logging
                self.global_step += 1
                
                # Add step counter and learning rate to loss dict
                loss_dict["global_step"] = self.global_step
                loss_dict["epoch"] = self.global_step / self.total_steps
                loss_dict["learning_rate"] = self.scheduler.get_last_lr()[0]
                
                # Log metrics
                should_log = (
                    self.global_step % self.log_every == 0 or  # Regular logging interval
                    self.global_step == 1 or  # First step
                    self.global_step == self.total_steps  # Last step
                )
                
                if should_log:
                    self.log(loss_dict)
                
                # Save periodically
                if (self.global_step % self.cfg["save_every"] == 0 or 
                    self.global_step == self.total_steps):
                    print(f"\nSaving checkpoint at step {self.global_step}")
                    self.save()
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        finally:
            # Final save
            print("\nSaving final checkpoint...")
            self.save()
            
            if self.use_wandb:
                print("Finishing wandb run...")
                wandb.finish()
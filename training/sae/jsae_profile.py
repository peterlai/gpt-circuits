"""
Profile Jacobian Sparse Autoencoder Training.
Based on concurrent training structure.

Usage: python -m training.sae.jsae_profile [options]
"""
# %%
import argparse
import os
import sys
from pathlib import Path
from functools import partial
import traceback # For detailed error printing

import torch
import torch.profiler
import einops

# Assuming these imports are correct based on previous context
from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.jsaesparsified import JSparsifiedGPT
from models.sparsified import SparsifiedGPTOutput
from training.sae import SAETrainer # Base class
from training.sae.concurrent import ConcurrentTrainer # Inherited class
from models.sae import SparseAutoencoder
from models.gpt import MLP

#@torch.compile # Keep commented out during profiling debugging
def get_jacobian_mlp_sae(
    sae_mlpin : SparseAutoencoder,
    sae_mlpout : SparseAutoencoder,
    mlp: MLP,
    topk_indices_mlpin: torch.Tensor,
    topk_indices_mlpout: torch.Tensor,
    mlp_act_grads: torch.Tensor,
) -> torch.Tensor:
    """Calculates the Jacobian block based on SAE features."""
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)

    dtype = wd1.dtype

    jacobian = einops.einsum(
        wd1[topk_indices_mlpin],
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_mlpout],
        # Einsum for Jacobian block calculation
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    )
    return jacobian


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments. Uses defaults relevant for profiling.
    """
    parser = argparse.ArgumentParser(description="Profile JSAE Training")
    # Keep original args for flexibility, but defaults might be less relevant now
    parser.add_argument("--config", type=str, default="jsae.shakespeare_64x4", help="Training config name (e.g., jsae.shakespeare_64x4)")
    parser.add_argument("--load_from", type=str, default="shakespeare_64x4", help="Base GPT model weights to load (e.g., shakespeare_64x4)")
    parser.add_argument("--name", type=str, default="PROF_RUN", help="Run name prefix for output dir (profiler logs go to ./tensorboard)")
    parser.add_argument("--sparsity", type=float, default=0.02, help="Jacobian sparsity loss coefficient")
    # Add specific profiling args if needed later
    return parser.parse_args()

class JSaeProfilerTrainer(ConcurrentTrainer):
    """
    Trainer class specifically for profiling JSAE.
    Inherits from ConcurrentTrainer and adds profiling hooks.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Initialize the JSAE model, load weights, and set up profiling.
        """
        print("Initializing JSaeProfilerTrainer...")
        # 1. Create Model (JSAE specific)
        model = JSparsifiedGPT(config.sae_config,
                               config.loss_coefficients,
                               config.trainable_layers)

        # 2. Load GPT Weights (JSAE specific)
        print(f"Loading base GPT weights from: {load_from}")
        model.load_gpt_weights(load_from)

        # 3. Freeze GPT Parameters (JSAE specific)
        print("Freezing GPT parameters.")
        for param in model.gpt.parameters():
            param.requires_grad = False

        # 4. Call Parent Initializer (Handles optimizer, scheduler, dataset etc.)
        # Use SAETrainer.__init__ as ConcurrentTrainer might not have its own __init__
        # Or directly call super().__init__(model, config) if ConcurrentTrainer handles it
        SAETrainer.__init__(self, model, config)
        print("Parent SAETrainer initialized.")

        # 5. Profiling Setup
        self.output_loss_calls = 0
        # Exit shortly after the first profiling cycle completes (1+1+3=5 steps)
        self.exit_after_calls = 10
        print(f"Profiling run configured to exit after {self.exit_after_calls} calls to output_to_loss.")

        self.profiler_log_dir = Path("./tensorboard") # As requested
        print(f"Profiler logs will be written to: {self.profiler_log_dir.resolve()}")

        # Ensure the log directory exists (with basic error handling)
        try:
            self.profiler_log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Ensured profiler log directory exists: {self.profiler_log_dir.resolve()}")
        except Exception as e:
            print(f"\n--- WARNING: Could not create log directory {self.profiler_log_dir.resolve()} ---")
            print(f"Error: {e}")
            print(f"Ensure the directory exists and has correct permissions.\n")

        # Prepare the trace handler
        standard_tb_handler = torch.profiler.tensorboard_trace_handler(
            str(self.profiler_log_dir),
            worker_name=f"rank_{self.rank}" if self.ddp else None # Add worker name if DDP
        )

        def custom_trace_handler_wrapper(log_dir, profiler_instance):
            """Wrapper to print message before writing trace."""
            step_num = self.output_loss_calls
            rank_info = f" (Rank {self.rank})" if self.ddp else ""
            print(f"\n--- Profiler{rank_info}: Attempting to write trace to {log_dir.resolve()} (after step {step_num}) ---")
            try:
                standard_tb_handler(profiler_instance)
                print(f"--- Profiler{rank_info}: Successfully called standard_tb_handler (after step {step_num}). Check directory. ---")
            except Exception as e:
                print(f"\n--- !!! ERROR{rank_info} during profiler trace writing (after step {step_num}) !!! ---")
                print(f"Target directory: {log_dir.resolve()}")
                print(f"Error type: {type(e)}")
                print(f"Error message: {e}")
                traceback.print_exc()
                print(f"--- !!! End of error details !!! ---\n")

        # Use partial to pass the log_dir to our wrapper
        self.profiler_trace_handler = partial(custom_trace_handler_wrapper, self.profiler_log_dir)

        if self.ddp:
            print("INFO: DDP detected. Profiler handler includes rank info.")
            # Dynamo DDP optimization might interfere with profiling/hooks
            torch._dynamo.config.optimize_ddp = False
            print("INFO: Set torch._dynamo.config.optimize_ddp = False")


    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Calculate loss from model output, including Jacobian term, wrapped in profiler.
        """
        self.output_loss_calls += 1
        print(f"DEBUG: output_to_loss call #{self.output_loss_calls}") # Debug print

        # --- Profiler Context ---
        # Schedule: Wait 1, Warmup 1, Active 3, Repeat 1
        # Options: Start with memory/stack profiling off due to previous issues
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=self.profiler_trace_handler,
            record_shapes=True,
            profile_memory=False, # Start with False
            with_stack=False     # Start with False
        ) as prof:
            # --- Core JSAE Loss Calculation ---
            # 1. Reconstruction Losses (from base model output)
            recon_losses = output.sae_losses # Shape: (2 * n_layers,)

            # 2. Jacobian Losses (JSAE specific)
            jacobian_losses = torch.empty(len(self.model.layer_idxs), device=recon_losses.device)
            for layer_idx in self.model.layer_idxs:
                # Extract necessary tensors from output
                topk_indices_mlpin = output.indices[f'{layer_idx}_mlpin']
                topk_indices_mlpout = output.indices[f'{layer_idx}_mlpout']
                mlp_act_grads = output.activations[f"{layer_idx}_mlpactgrads"]

                # Get relevant model components
                sae_mlpin = self.model.saes[f'{layer_idx}_mlpin']
                sae_mlpout = self.model.saes[f'{layer_idx}_mlpout']
                mlp = self.model.gpt.transformer.h[layer_idx].mlp
                k = sae_mlpin.k # Assuming k is same for mlpin/mlpout

                # Calculate Jacobian block
                jacobian = get_jacobian_mlp_sae(
                    sae_mlpin=sae_mlpin,
                    sae_mlpout=sae_mlpout,
                    mlp=mlp,
                    topk_indices_mlpin=topk_indices_mlpin,
                    topk_indices_mlpout=topk_indices_mlpout,
                    mlp_act_grads=mlp_act_grads,
                )

                # Calculate Jacobian sparsity loss
                j_coeff = self.model.loss_coefficients.sparsity[layer_idx]
                jacobian_loss = j_coeff * torch.abs(jacobian).sum() / (k ** 2)
                jacobian_losses[layer_idx] = jacobian_loss

            output.recon_losses = recon_losses
            output.sparsity_losses = jacobian_losses
            
            pair_recon_loss = einops.rearrange(recon_losses, "(pair loc) -> pair loc", pair=2).sum(dim=-1)
            losses = pair_recon_loss + jacobian_losses
            # --- Profiler Step (handled by schedule) ---
            # If using a schedule, prof.step() is called automatically by the context manager.
            # print(f"DEBUG: Inside profiler context step {self.output_loss_calls}") # Optional debug

        # --- Check Exit Condition ---
        # Check *after* the profiler block to ensure the step completes
        if self.output_loss_calls >= self.exit_after_calls:
            print(f"\n--- Reached {self.output_loss_calls} calls to output_to_loss. Exiting script. ---")
            sys.exit(0) # Exit gracefully

        return losses

    # gather_metrics can likely be inherited directly from SAETrainer/ConcurrentTrainer
    # def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
    #     """ Gather metrics from loss and model output. """
    #     return super().gather_metrics(loss, output)


if __name__ == "__main__":
    print("Starting JSAE Profiling Script...")
    # Parse command line arguments
    args = parse_args()
    print(f"Arguments: {args}")

    # Load configuration
    config_name = args.config
    try:
        config: SAETrainingConfig = options[config_name]
        print(f"Loaded config: {config_name}")
    except KeyError:
        print(f"ERROR: Config '{config_name}' not found in options.")
        sys.exit(1)

    # Apply command-line overrides to config
    if args.name:
        config.name = args.name
        print(f"Set run name to: {config.name}")

    if args.sparsity is not None: # Check if sparsity was actually provided
        num_trainable_layers = len(config.loss_coefficients.sparsity)
        if num_trainable_layers > 0:
            config.loss_coefficients.sparsity = (args.sparsity,) * num_trainable_layers
            print(f"Set sparsity coefficient to {args.sparsity} for all {num_trainable_layers} layers")
            # Optionally update name based on sparsity
            config.name = f"{config.name}-sparsity-{args.sparsity:.1e}"
            print(f"Updated run name to: {config.name}")
        else:
            print("Warning: Sparsity argument provided, but no trainable layers found in config.")

    # Determine GPT model path
    gpt_load_path = TrainingConfig.checkpoints_dir / args.load_from
    print(f"GPT model path: {gpt_load_path}")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = JSaeProfilerTrainer(config, load_from=gpt_load_path)

    # Run training (will be interrupted by sys.exit in output_to_loss)
    print("Starting training loop (will exit early for profiling)...")
    try:
        trainer.train()
    except SystemExit:
        # Catch the planned exit
        print(f"\nTraining stopped early after {trainer.output_loss_calls} calls as planned for profiling.")
        print("Profiler trace (if generated) should be in ./tensorboard.")
    except Exception as e:
        print("\n--- !!! An unexpected error occurred during training !!! ---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        traceback.print_exc()
        print("--- !!! End of error details !!! ---\n")
        sys.exit(1) # Exit with error code

    print("\nJSAE Profiling Script finished.")
# %%

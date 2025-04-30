"""
python -m training.sae.jsae_concurrent [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4] [--sparsity=0.02|0.1,0.2,0.3,0.4]
"""
# %%
import argparse
import os
import sys
from pathlib import Path

import einops
import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.gpt import MLP
from models.jsaesparsified import JSparsifiedGPT
from models.sae import SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
from training.sae import SAETrainer
from training.sae.concurrent import ConcurrentTrainer

import dataclasses
import json
from config.sae.models import SAEConfig

from typing import Optional, List, Union
# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())

from utils.jsae import jacobian_mlp

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config", default="jsae.shakespeare_64x4")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load", default="shakespeare_64x4")
    parser.add_argument("--name", type=str, help="Model name for checkpoints", default="jsae.shk_64x4")
    parser.add_argument("--sparsity", type=str, help="Jacobian sparsity loss coefficient(s). Either a single float or comma-separated floats for each layer.", default="0.02")
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps to train", default=7500)
    return parser.parse_args()


class JSaeTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        model = JSparsifiedGPT(config.sae_config, 
                                  config.loss_coefficients, 
                                  config.trainable_layers)

        # Load GPT weights
        model.load_gpt_weights(load_from)

        # Freeze GPT parameters
        for param in model.gpt.parameters():
            param.requires_grad = False

        SAETrainer.__init__(self, model, config)

        if self.ddp:
            # HACK: We're doing something that causes DDP to crash unless DDP optimization is disabled.
            torch._dynamo.config.optimize_ddp = False  # type: ignore
            
    def save_checkpoint(self, model: JSparsifiedGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # As weights are shared, only save if each layer is better than the previous best.
        dir = self.config.out_dir
        
        self.model.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = dataclasses.asdict(self.config.sae_config, dict_factory=SAEConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        for layer_idx in self.model.layer_idxs:
            if is_best[layer_idx]:
            # save this pair of saes!
                for loc in ['mlpin', 'mlpout']:
                    sae = self.model.saes[f'{layer_idx}_{loc}']
                    assert "jsae" in sae.config.sae_variant
                    sae.save(Path(dir))

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Calculate model loss.
        """
        output: SparsifiedGPTOutput = self.model(x, y, is_eval=is_eval)
        loss = self.output_to_loss(output, is_eval=is_eval)
        metrics = None

        # Only include metrics if in evaluation mode
        if is_eval:
            metrics = self.gather_metrics(loss, output)

        return loss, metrics

    # TODO: This is a very expensive operation, we should try to speed it up
    def output_to_loss(self, output: SparsifiedGPTOutput, is_eval: bool= False) -> torch.Tensor:
        """
        Return an array of losses, one for each pair of layers:
        loss[i] = loss_recon[f"{i}_mlpin"] + loss_recon[f"{i}_mlpout"] 
            + l1_jacobian(f"{i}_mlpout", f"{i}_mlpin")
        """
        device = output.sae_losses.device
        recon_losses = output.sae_losses
        jacobian_losses = torch.zeros(len(self.model.layer_idxs), device=device)
        j_coeffs = torch.tensor(self.model.loss_coefficients.sparsity, device=device)
        
        
        for layer_idx in self.model.layer_idxs:
            if self.model.loss_coefficients.sparsity[layer_idx] == 0 and not is_eval: # compute jacobian loss only on eval if sparsity is 0
                continue
            topk_indices_mlpin = output.indices[f'{layer_idx}_mlpin']
            topk_indices_mlpout = output.indices[f'{layer_idx}_mlpout']

            mlp_act_grads = output.activations[f"{layer_idx}_mlpactgrads"]

            jacobian_loss = jacobian_mlp(
                sae_mlpin = self.model.saes[f'{layer_idx}_mlpin'],
                sae_mlpout = self.model.saes[f'{layer_idx}_mlpout'],
                mlp = self.model.gpt.transformer.h[layer_idx].mlp,
                topk_indices_mlpin = topk_indices_mlpin,
                topk_indices_mlpout = topk_indices_mlpout,
                mlp_act_grads = mlp_act_grads,
            )

            # Each SAE has it's own loss term, and are trained "independently"
            # so we will put the jacobian loss into the aux loss term
            # for the sae_mlpout for each pair of SAEs
            jacobian_losses[layer_idx] = jacobian_loss

        # Store computed loss components in sparsify output to be read out by gather_metrics
        output.sparsity_losses = jacobian_losses.detach()

        pair_losses = einops.rearrange(recon_losses, "(layer pair) -> layer pair", pair=2).sum(dim=-1)
        losses = pair_losses + j_coeffs * jacobian_losses # (layer)
        return losses


    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics =  super().gather_metrics(loss, output)
        metrics["∇_l1"] = output.sparsity_losses
        metrics["recon_l2"] = output.recon_losses

        return metrics

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    if args.name:
        config.name = args.name
    if args.max_steps:
        config.max_steps = args.max_steps

    if args.sparsity is not None:
        try:
            sparsity_values_str = args.sparsity.split(',')
            sparsity_values = [float(s.strip()) for s in sparsity_values_str]
        except ValueError:
            print(f"Error: Invalid format for --sparsity. Expected a single float or comma-separated floats. Got: {args.sparsity}")
            sys.exit(1)

        num_trainable_layers = len(config.loss_coefficients.sparsity) # Use initial config to know layer count

        if len(sparsity_values) == 1:
            # Single value provided, apply to all layers
            sparsity_val = sparsity_values[0]
            config.loss_coefficients.sparsity = (sparsity_val,) * num_trainable_layers
            print(f"Setting sparsity to {sparsity_val} for all {num_trainable_layers} layers")
            config.name = f"{config.name}-sparse-{sparsity_val:.1e}"
        elif len(sparsity_values) == num_trainable_layers:
            # Correct number of values provided for each layer
            config.loss_coefficients.sparsity = tuple(sparsity_values)
            print(f"Setting per-layer sparsity: {config.loss_coefficients.sparsity}")
            # Modify name to include all sparsity values
            sparsity_str = "_".join([f"{s:.1e}" for s in sparsity_values])
            config.name = f"{config.name}-sparse-{sparsity_str}"
        else:
            # Incorrect number of values
            raise ValueError(f"Error: --sparsity must provide either 1 value (for all layers) or {num_trainable_layers} values (one per layer). Got {len(sparsity_values)} values.")

    # Initialize trainer
    trainer = JSaeTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    print(f'{trainer.model.saes.keys()=}')
    print(f'{trainer.model.layer_idxs=}')
    print(f'Using sparsity coefficients: {trainer.model.loss_coefficients.sparsity}')
    trainer.train()

# %%

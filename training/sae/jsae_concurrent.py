"""
python -m training.sae.jsae_concurrent [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4] [--sparsity=0.02]
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

# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())

@torch.compile(mode="max-autotune", fullgraph=True)
def get_jacobian_mlp_sae(
    sae_mlpin : SparseAutoencoder,
    sae_mlpout : SparseAutoencoder,
    mlp: MLP,
    topk_indices_mlpin: torch.Tensor,
    topk_indices_mlpout: torch.Tensor,
    mlp_act_grads: torch.Tensor,
) -> torch.Tensor:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)
    
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)

    dtype = wd1.dtype
    k = sae_mlpin.k
    jacobian = einops.einsum(
        wd1[topk_indices_mlpin],
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_mlpout],
        # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
        # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    ).abs_().sum() / (k ** 2)
    return jacobian

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config", default="jsae.shakespeare_64x4")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load", default="shakespeare_64x4")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    parser.add_argument("--sparsity", type=float, help="Jacobian sparsity loss coefficient", default=0.02)
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

    # TODO: This is a very expensive operation, we should try to speed it up
    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Return an array of losses, one for each pair of layers:
        loss[i] = loss_recon[f"{i}_mlpin"] + loss_recon[f"{i}_mlpout"] 
            + l1_jacobian(f"{i}_mlpout", f"{i}_mlpin")
        """
        device = output.sae_losses.device
        recon_losses = output.sae_losses
        jacobian_losses = torch.empty(len(self.model.layer_idxs), device=device)
        j_coeffs = torch.tensor(self.model.loss_coefficients.sparsity, device=device)
        
        for layer_idx in self.model.layer_idxs:
            topk_indices_mlpin = output.indices[f'{layer_idx}_mlpin']
            topk_indices_mlpout = output.indices[f'{layer_idx}_mlpout']
            
            mlp_act_grads = output.activations[f"{layer_idx}_mlpactgrads"]
            
            jacobian_loss = get_jacobian_mlp_sae(
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
        
    if args.sparsity:
        num_trainable_layers = len(config.loss_coefficients.sparsity)
        config.loss_coefficients.sparsity = (args.sparsity, ) * num_trainable_layers
        print(f"Setting sparsity to {args.sparsity} for all {num_trainable_layers} layers")
        
        config.name = f"{config.name}-sparsity-{args.sparsity:.1e}"
        
        # Update outdir


    # Initialize trainer
    trainer = JSaeTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

# %%

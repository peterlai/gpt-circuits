"""
python -m training.sae.mlp_concurrent --config=mlp-topk.shakespeare_64x4 --load_from=shakespeare_64x4
"""
# %%
import argparse
from pathlib import Path

import sys
import os

# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())


import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, shakespeare_64x4_defaults

from config.sae.models import SAEConfig, SAEVariant
from config.gpt.models import gpt_options
from config.sae.training import options
from config.sae.training import LossCoefficients

from models.mlpsparsified import MLPSparsifiedGPT

from training.sae import SAETrainer
from training.sae.concurrent import ConcurrentTrainer
from training import Trainer

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()

class MLPConcurrentTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        model = MLPSparsifiedGPT(config.sae_config, 
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
            
    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
        # Train weights.
        Trainer.train(self)
        # Wait for all processes to complete training.
        if self.ddp:
            torch.distributed.barrier()

        # Reload all checkpoint weights, which may include those that weren't trained.
        self.model = MLPSparsifiedGPT.load(
            self.config.out_dir,
            loss_coefficients=self.config.loss_coefficients,
            trainable_layers=None,  # Load all layers
            device=self.config.device,
        ).to(self.config.device)
        # Wrap the model if using DDP
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])  # type: ignore

        # Gather final metrics. We don't bother compiling because we're just running eval once.
        final_metrics = self.val_step(0, should_log=False)  # step 0 so checkpoint isn't saved.
        self.checkpoint_l0s = final_metrics["l0s"]
        self.checkpoint_ce_loss = final_metrics["ce_loss"]
        self.checkpoint_ce_loss_increases = final_metrics["ce_loss_increases"]
        self.checkpoint_compound_ce_loss_increase = final_metrics["compound_ce_loss_increase"]

        # Summarize results
        if self.is_main_process:
            print(f"Final L0s: {self.pretty_print(self.checkpoint_l0s)}")
            print(f"Final CE loss increases: {self.pretty_print(self.checkpoint_ce_loss_increases)}")
            print(f"Final compound CE loss increase: {self.pretty_print(self.checkpoint_compound_ce_loss_increase)}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Update outdir
    if args.name:
        config.name = args.name

    # Initialize trainer
    trainer = MLPConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

# %%

"""
Train SAE weights for all layers concurrently.

$ python -m david.jsae.train
"""
# %%
import argparse
from pathlib import Path

import sys
import os

# Change current working directory to parent
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())


import torch


from config import TrainingConfig
from config.sae.training import SAETrainingConfig, shakespeare_64x4_defaults
from models.sparsified import SparsifiedGPTOutput
from config.sae.models import SAEConfig, SAEVariant
from config.gpt.models import gpt_options
from config.sae.training import LossCoefficients
from david.jsae.sparsified import JSAESparsifiedGPT
from training.sae import SAETrainer
from training.sae.concurrent import ConcurrentTrainer

class JSAEConcurrentTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        model = JSAESparsifiedGPT(config.sae_config, 
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


# %%
if __name__ == "__main__":
    # Parse command line arguments
  
    # Load configuration
    config = SAETrainingConfig(
        name="mlp-topk.shakespeare_64x4",
        sae_config=SAEConfig(
            name="topk-10-x8-mlp.shakespeare_64x4",
            gpt_config=gpt_options["ascii_64x4"],
            n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
            top_k=(10,10,10,10,10,10,10,10),
            sae_variant=SAEVariant.TOPK,
        ),
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients()
    )

    # Initialize trainer
    trainer = JSAEConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "shakespeare_64x4")
    trainer.train()
# %%

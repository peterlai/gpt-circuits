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
from training import Trainer
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
        self.model = JSAESparsifiedGPT.load(
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



# %%
if __name__ == "__main__":
    # Parse command line arguments
    mlp_sae_defaults = {
    "data_dir": "data/shakespeare",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "warmup_steps": 750,
    "max_steps": 100,
    "decay_lr": True,
    "min_lr": 1e-4,
}
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
        **mlp_sae_defaults,
        loss_coefficients=LossCoefficients()
    )

    # Initialize trainer
    trainer = JSAEConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "shakespeare_64x4")
    trainer.train()
# %%

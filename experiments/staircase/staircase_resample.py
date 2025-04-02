"""
Train SAE weights for all layers concurrently, and resamples the biases every eval interval.

$ python -m training.sae.staircase_sequential --config=topk-staircase-detach.shakespeare_64x4 --load_from=shakespeare_64x4
    $ torchrun --standalone --nproc_per_node=8 -m training.sae.staircase_concurrent --config=topk-staircase-share.stories_256x4 --load_from=stories_256x4
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from config import TrainingConfig
from config.sae.training import options
from training.sae.staircase_concurrent import StaircaseConcurrentTrainer
from training.sae import SAETrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class StaircaseResampleTrainer(StaircaseConcurrentTrainer):
    """
    Train SAE weights for all layers sequentially.
    """

    def resample(self, step):
        """
        Resample biases.
        """
        if step < 1000:
            keys = ['1', '2', '3', '4']
        elif step < 2500:
            keys = ['3', '4']
        elif step < 3500:
            keys = ['4']
        else:
            keys = []
        
        for key in keys:
            print(f"Resampling {key} ...", end="")
            self.model.saes[key].resample()
            print("Resample done")

    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
        # Train weights.
         # Prepare directory for checkpoints
        if self.is_main_process:
            os.makedirs(self.config.out_dir, exist_ok=True)

            # Print configuration
            self.log(dataclasses.asdict(self.config), self.LogDestination.DEBUG)

        # Set the float32 matmul precision to high for better performance.
        torch.set_float32_matmul_precision("high")

        # Let's see what we're starting with.
        self.val_step(0)

        # Start training.
        for step in range(1, self.config.max_steps + 1):
            self.train_step(step)

            # Always evaluate the model at the end of training.
            last_step = step == self.config.max_steps
            if step % self.config.eval_interval == 0 or last_step:
                self.val_step(step)
                # use eval interval to resample biases
                self.resample(step)
                
    
        SAETrainer.train(self)

if __name__ == "__main__":
    # Parse command line arguments
    #args = parse_args()

    # Load configuration
    #config_name = args.config
    config = options["topk-staircase-detach.shakespeare_64x4"]
    assert "staircase" in config.sae_config.sae_variant, "Staircase trainer must use staircase SAE variant"
    # Update outdir
    # if args.name:
    #     config.name = args.name
    load_from = "shakespeare_64x4"
    # Initialize trainer
    trainer = StaircaseResampleTrainer(config, load_from=TrainingConfig.checkpoints_dir / load_from)
    trainer.train()

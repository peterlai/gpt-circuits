"""
Train Staircase SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.

$ python -m experiments.staircase.training
"""

import argparse

import torch

from config import TrainingConfig
from config.sae.models import sae_options
from config.sae.training import LossCoefficients, SAETrainingConfig
from models.sparsified import SparsifiedGPTOutput
from training.sae.end_to_end import EndToEndTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from", type=str, default="shakespeare_64x4", help="GPT model weights to load")
    parser.add_argument("--name", type=str, default="e2e.jumprelu-staircase.shakespeare_64x4", help="Model name for checkpoints")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = SAETrainingConfig(
        name=args.name,
        sae_config=sae_options["jumprelu-staircase-x8.shakespeare_64x4"],
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=750,
        max_steps=7500,
        decay_lr=True,
        min_lr=1e-4,
        loss_coefficients=LossCoefficients(
            sparsity=(0.002, 0.003, 0.01, 0.01, 0.1), # l0s
            downstream=1.0,
            bandwidth=0.1,
        ),
    )

    # Initialize trainer
    trainer = EndToEndTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()


if __name__ == "__main__":
    main()
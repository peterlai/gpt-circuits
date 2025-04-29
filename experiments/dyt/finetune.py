"""
Finetune GPT model:
$ python -m experiments.dyt.finetune --load_from=shakespeare_64x4 --config=shakespeare_64x4_dyt
"""

import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_model

from config import TrainingConfig
from config.gpt.models import GPTConfig, NormalizationStrategy
from config.gpt.training import options
from models.gpt import GPT
from training import Trainer  # Ensure Trainer is imported
from training.gpt import GPTTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--config", type=str, help="Training config to use for fine-tuning")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class DynamicTanFineTuner(GPTTrainer):
    """
    Fine-tunes a GPT model from an existing checkpoint using dynamic tanh instead of layer norm.
    """

    def __init__(self, load_from: Path, training_config: TrainingConfig):
        """
        Load GPT model.
        """
        meta_path = os.path.join(load_from, "model.json")
        weights_path = os.path.join(load_from, "model.safetensors")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        gpt_config = GPTConfig(**meta)
        gpt_config.norm_strategy = NormalizationStrategy.DYNAMIC_TANH

        model = GPT(gpt_config)
        # Using strict=False to allow loading weights even if the model architecture has changed
        load_model(model, weights_path, device=training_config.device.type, strict=False)

        # Freeze all non-dynamic tanh parameters
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ["mlp", "alpha", "beta", "gamma"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Call Trainer's __init__ directly, skipping GPTTrainer's __init__
        Trainer.__init__(self, model, training_config)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    training_config_name = args.config
    training_config = options[training_config_name]

    # Update outdir if name is provided
    if args.name:
        training_config.name = args.name

    # Initialize trainer
    trainer = DynamicTanFineTuner(
        load_from=TrainingConfig.checkpoints_dir / args.load_from,
        training_config=training_config,
    )
    trainer.train()

    # Pring final result
    if trainer.is_main_process:
        print(f"Best validation loss: {round(trainer.best_val_loss.item(), 4)}")

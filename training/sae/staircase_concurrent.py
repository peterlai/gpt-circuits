"""
Train SAE weights for all layers concurrently.

$ python -m training.sae.staircase_concurrent --config=topk-staircase-share.shakespeare_64x4 --load_from=shakespeare_64x4
$ torchrun --standalone --nproc_per_node=8 -m training.sae.staircase_concurrent --config=topk-staircase-share.stories_256x4 --load_from=stories_256x4
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from config import TrainingConfig
from config.sae.models import SAEConfig
from config.sae.training import SAETrainingConfig, options
from models.sae import SparseAutoencoder
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from models.sae.topk import StaircaseTopKSharedContext, StaircaseTopKSAE
from training.sae.concurrent import ConcurrentTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class StaircaseConcurrentTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def save_checkpoint(self, model: SparsifiedGPT, is_best: torch.Tensor):
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
        
        if torch.all(is_best):
            # Save SAE modules
            for module in self.model.saes.values():
                assert isinstance(module, StaircaseTopKSAE)
                module.save(Path(dir))


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]
    assert "staircase" in config.sae_config.sae_variant, "Staircase trainer must use staircase SAE variant"
    # Update outdir
    if args.name:
        config.name = args.name

    # Initialize trainer
    trainer = StaircaseConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

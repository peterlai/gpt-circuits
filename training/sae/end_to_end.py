"""
Train SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.

$ python -m training.sae.end_to_end --config=end-to-end.shakespeare_64x4 --load_from=shakespeare_64x4
$ torchrun --standalone --nproc_per_node=8 -m training.sae.end_to_end --config=e2e.jumprelu.tiny_256x4 --load_from=tiny_256x4
"""

import argparse
from contextlib import contextmanager
from typing import Iterable

import torch

from config import TrainingConfig
from config.sae.training import options
from models.sparsified import SparsifiedGPTOutput
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


class EndToEndTrainer(ConcurrentTrainer):
    """
    Train SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.
    https://arxiv.org/pdf/2405.12241
    """

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Return a stack of end-to-end losses, one for each trainable layer.
        """
        losses = []
        for layer_idx in output.sae_loss_components.keys():
            e2e_loss = self.calculate_e2e_loss(output, layer_idx)
            losses.append(e2e_loss)

        return torch.stack(losses)

    def calculate_e2e_loss(self, output: SparsifiedGPTOutput, layer_idx: int) -> torch.Tensor:
        """
        Calculate end-to-end loss for a single layer.

        :param output: Sparsified model output.
        :param layer_idx: Index of the layer to calculate loss for.
        """
        target_logits = output.logits
        vocab_size = output.logits.size(-1)
        n_blocks = len(self.unwrapped_model.gpt.transformer.h)
        reconstructed_activations = output.reconstructed_activations[layer_idx]

        # Forward pass with reconstructed activations
        target_layers = list(range(layer_idx + 1, n_blocks))  # Never target the last layer
        with self.collect_activations(target_layers=target_layers) as predicted_activations:
            predicted_logits = self.unwrapped_model.gpt.forward_with_patched_activations(reconstructed_activations, layer_idx)

        # Caculate downstream reconstruction loss
        downstream_losses = []
        assert self.config.loss_coefficients.downstream is not None, "Downstream loss coefficient must be set"
        for downstream_layer_idx, x_predicted in predicted_activations.items():
            x = output.activations[downstream_layer_idx]
            downstream_loss = (x - x_predicted).pow(2).sum(dim=-1).mean()
            downstream_loss *= self.config.loss_coefficients.downstream  # Scale by downstream loss coefficient
            downstream_losses.append(downstream_loss)
        downstream_loss = torch.stack(downstream_losses).mean() if downstream_losses else torch.tensor(0.0)

        # Calculate KL divergence between target and reconstructed logits
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(predicted_logits, dim=-1).view(-1, vocab_size),
            torch.nn.functional.softmax(target_logits, dim=-1).view(-1, vocab_size),
            reduction="batchmean",
        )

        # Reuse original sparsity loss
        sparsity_loss = output.sae_loss_components[layer_idx].sparsity

        return kl_div + sparsity_loss + downstream_loss

    @contextmanager
    def collect_activations(self, target_layers: Iterable[int] = ()):
        """
        Context manager for collecting activations during the forward pass.

        :param target_layers: Target layer indices. 1 means we collect activations just after the first transformer block.
        :yield activations: Dictionary of activations.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in target_layers:
            assert layer_idx > 0, "Must target activations after the first transformer block"
            target = self.unwrapped_model.gpt.transformer.h[layer_idx - 1]
            hook = self.create_hook(activations, layer_idx)
            hooks.append(target.register_forward_hook(hook))

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_hook(self, activations, layer_idx):
        """
        Create a forward hook that records activations after the forward pass.

        :param activations: List for storing activations.
        :param layer_idx: Index of the layer to record activations for.
        """

        def hook(_, input, output):
            activations[layer_idx] = output

        return hook


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
    trainer = EndToEndTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

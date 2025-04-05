from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn

from models.sae import EncoderOutput
from models.sparsified import SparsifiedGPT


class SparsifiedMLPGPT(SparsifiedGPT):
    """
    GPT Model with sparsified activations across MLP modules using sparse autoencoders.
    """

    @contextmanager
    def record_activations(self):
        """
        Context manager for recording the MLP inputs and outputs.

        :yield activations: Dictionary of activations for each SAE layer.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in list(range(len(self.config.n_features))):
            target = self.get_hook_target(layer_idx)
            hook = self.create_activation_hook(activations, layer_idx)
            hooks.append(target.register_forward_hook(hook))  # type: ignore

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_activation_hook(self, activations, layer_idx):
        """
        Create a forward hook for the given layer index for recording activations.

        :param activations: Dictionary for storing activations.
        :param layer_idx: Layer index to record activations for.
        """

        def activation_hook(_, inputs, output):
            # even layer_idx are input to MLP, odd are MLP output
            # h[0].mlp_in, h[0].mlp_out, h[1].mlp_in, h[1].mlp_out, ...
            if layer_idx % 2 == 0:
                activations[layer_idx] = inputs[0]
            else:
                activations[layer_idx] = output

        return activation_hook

    @contextmanager
    def use_saes(self, layers_to_patch: Iterable[int] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param layers_to_patch: Layer indices for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        """
        # Dictionary for storing results
        encoder_outputs: dict[int, EncoderOutput] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            target = self.get_hook_target(layer_idx)
            sae = self.saes[f"{layer_idx}"]
            # Output values will be overwritten (hack to pass object by reference)
            output = EncoderOutput(torch.tensor(0), torch.tensor(0))
            should_patch_activations = layer_idx in layers_to_patch
            if layer_idx % 2 == 0:
                hook = self.create_sae_pre_hook(sae, output, should_patch_activations)
                hooks.append(target.register_forward_pre_hook(hook))  # type: ignore
            else:
                hook = self.create_sae_post_hook(sae, output, should_patch_activations)
                hooks.append(target.register_forward_hook(hook))  # type: ignore
            encoder_outputs[layer_idx] = output

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_sae_post_hook(self, sae, sae_output, should_patch_activations):
        """
        Create a forward post-hook for the given layer index for applying sparse autoencoding.

        :param sae: SAE module to use for the forward pass.
        :param sae_output: Encoder output to be updated.
        :param should_patch_activations: Whether to patch activations.
        """

        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_hook(_, inputs, output):
            """
            NOTE: Compiling seems to struggle with branching logic, and so we disable it (non-recursively).
            """

            x = output
            # Override field values instead of replacing reference
            sae_output.__dict__ = sae(x).__dict__

            # Patch activations if needed
            return (
                output.reconstructed_activations if should_patch_activations else output
            )

        return sae_hook

    def get_hook_target(self, layer_idx: int) -> nn.Module:
        """
        SAE layer -> Targeted module for forward hook.

        Every pair of SAE layers target the same MLP, representing either the inputs or outputs.
        E.g.: Layers 0 and 1 both target the first MLP.
        """
        return self.gpt.transformer.h[layer_idx // 2].mlp  # type: ignore

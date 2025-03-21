import dataclasses
import json
import os
from contextlib import contextmanager
from typing import Iterable, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.nn import functional as F

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SAELossComponents
from models.sae.gated import GatedSAE, GatedSAE_V2
from models.sae.jumprelu import JumpReLUSAE
from models.sae.standard import StandardSAE, StandardSAE_V2
from models.sae.topk import TopKSAE


@dataclasses.dataclass
class SparsifiedGPTOutput:
    """
    Output from the forward pass of a sparsified GPT model.
    """

    logits: torch.Tensor
    cross_entropy_loss: torch.Tensor
    # Residual stream activations at every layer
    activations: dict[int, torch.Tensor]
    ce_loss_increases: Optional[torch.Tensor]
    # Compound cross-entropy loss increase if using SAE reconstructions for all trainable layers
    compound_ce_loss_increase: Optional[torch.Tensor]
    sae_loss_components: dict[int, SAELossComponents]
    feature_magnitudes: dict[int, torch.Tensor]
    reconstructed_activations: dict[int, torch.Tensor]

    @property
    def sae_losses(self) -> torch.Tensor:
        """
        SAE losses for each trainable layer.
        """
        return torch.stack([loss.total for loss in self.sae_loss_components.values()])


class SparsifiedGPT(nn.Module):
    """
    GPT Model with sparsified activations using sparse autoencoders.
    """

    def __init__(
        self,
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__()
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)

        # Construct sae layers
        sae_class = self.get_sae_class(config)
        self.layer_idxs = trainable_layers if trainable_layers else list(range(len(config.n_features)))
        self.saes = nn.ModuleDict(dict([(f"{i}", sae_class(i, config, loss_coefficients)) for i in self.layer_idxs]))

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, is_eval: bool = False
    ) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.

        :param idx: Input tensor.
        :param targets: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        """
        with self.record_activations() as activations:
            with self.use_saes() as encoder_outputs:
                logits, cross_entropy_loss = self.gpt(idx, targets)

        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE layer
            ce_loss_increases = []
            for layer_idx, output in encoder_outputs.items():
                x = output.reconstructed_activations
                sae_logits = self.gpt.forward_with_patched_activations(x, layer_idx)
                sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_saes(layers_to_patch=self.layer_idxs):
                _, compound_cross_entropy_loss = self.gpt(idx, targets)
                compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            activations=activations,
            ce_loss_increases=ce_loss_increases,
            compound_ce_loss_increase=compound_ce_loss_increase,
            sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
            feature_magnitudes={i: output.feature_magnitudes for i, output in encoder_outputs.items()},
            reconstructed_activations={i: output.reconstructed_activations for i, output in encoder_outputs.items()},
        )

    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in list(range(len(self.config.n_features))):
            target = self.get_hook_target(layer_idx)
            hook = self.create_activation_hook(activations, layer_idx)
            hooks.append(target.register_forward_pre_hook(hook))  # type: ignore

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_activation_hook(self, activations, layer_idx):
        """
        Create a forward pre-hook for the given layer index for recording activations.

        :param activations: Dictionary for storing activations.
        :param layer_idx: Layer index to record activations for.
        """

        def activation_hook(_, inputs):
            activations[layer_idx] = inputs[0]

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
            hook = self.create_sae_hook(sae, output, should_patch_activations)
            hooks.append(target.register_forward_pre_hook(hook))  # type: ignore
            encoder_outputs[layer_idx] = output

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_sae_hook(self, sae, output, should_patch_activations):
        """
        Create a forward pre-hook for the given layer index for applying sparse autoencoding.

        :param sae: SAE module to use for the forward pass.
        :param output: Encoder output to be updated.
        :param should_patch_activations: Whether to patch activations.
        """

        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_hook(_, inputs):
            """
            NOTE: Compiling seems to struggle with branching logic, and so we disable it (non-recursively).
            """

            x = inputs[0]
            # Override field values instead of replacing reference
            output.__dict__ = sae(x).__dict__

            # Patch activations if needed
            return (output.reconstructed_activations,) if should_patch_activations else inputs

        return sae_hook

    def get_hook_target(self, layer_idx) -> nn.Module:
        """
        SAE layer -> Targeted module for forward pre-hook.
        """
        if layer_idx < self.config.gpt_config.n_layer:
            return self.gpt.transformer.h[layer_idx]  # type: ignore
        elif layer_idx == self.config.gpt_config.n_layer:
            return self.gpt.transformer.ln_f  # type: ignore
        raise ValueError(f"Invalid layer index: {layer_idx}")

    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load GPT model
        gpt = GPT.load(dir, device=device)

        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)
        config.gpt_config = gpt.config

        # Create model using saved config
        model = SparsifiedGPT(config, loss_coefficients, trainable_layers)
        model.gpt = gpt

        # Load SAE weights
        for layer_name, module in model.saes.items():
            weights_path = os.path.join(dir, f"sae.{layer_name}.safetensors")
            load_model(module, weights_path, device=device.type)

        return model

    def load_gpt_weights(self, dir):
        """
        Load just the GPT model weights without loading SAE weights.
        """
        device = next(self.gpt.lm_head.parameters()).device
        self.gpt = GPT.load(dir, device=device)

    def save(self, dir, layers_to_save: Optional[list[str]] = None):
        """
        Save the sparsified GPT model to a directory.

        :param dir: Directory for saving weights.
        :param layers_to_save: Module names for SAE layers to save. If None, all layers will be saved.
        """
        # Save GPT model
        self.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = dataclasses.asdict(self.config, dict_factory=SAEConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Which layers should we save?
        layers_to_save = layers_to_save or list(self.saes.keys())

        # Save SAE modules
        for layer_name, module in self.saes.items():
            if layer_name in layers_to_save:
                weights_path = os.path.join(dir, f"sae.{layer_name}.safetensors")
                save_model(module, weights_path)

    def get_sae_class(self, config: SAEConfig) -> type:
        """
        Maps the SAE variant to the actual class.
        """
        match config.sae_variant:
            case SAEVariant.STANDARD:
                return StandardSAE
            case SAEVariant.STANDARD_V2:
                return StandardSAE_V2
            case SAEVariant.GATED:
                return GatedSAE
            case SAEVariant.GATED_V2:
                return GatedSAE_V2
            case SAEVariant.JUMP_RELU:
                return JumpReLUSAE
            case SAEVariant.TOPK:
                return TopKSAE
            case _:
                raise ValueError(f"Unrecognized SAE variant: {self.sae_variant}")

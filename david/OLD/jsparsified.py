import dataclasses
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.mlpgpt import MLP_GPT
from models.sae.jsae import JSAE
from models.mlpsparsified import MLPSparsifiedGPT

import itertools

def flatten(list2d):
    return list(itertools.chain(*list2d))


class MLPSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = MLP_GPT(config.gpt_config)
        assert len(config.n_features) == self.gpt.config.n_layer * 2
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in ['mlpin', 'mlpout']] # index of the mlpin and mlpout activations
        
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        
        self.saes = nn.ModuleDict(dict([(key, sae_class(idx, config, loss_coefficients, self)) 
                                        for idx, key in enumerate(sae_keys)]))

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
            for activation_idx, output in encoder_outputs.items():
                x = output.reconstructed_activations
                sae_logits = self.gpt.forward_with_patched_activations(x, activation_idx)
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
        return MLPSparsifiedGPT.record_activations(self)

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
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        """
        # Dictionary for storing results
        encoder_outputs: dict[str, tuple[EncoderOutput, EncoderOutput]] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            target = self.gpt.transformer.h[layer_idx].mlp
            should_patch_activations = layer_idx in activations_to_patch
            
            sae_mlpin = self.saes[f"{layer_idx}"].mlp_in
            sae_mlpout = self.saes[f"{layer_idx}"].mlp_out
            hook_fn = self.create_sae_hook(sae_mlpin, sae_mlpout, encoder_outputs, should_patch_activations)
            
            hooks.append(target.register_forward_pre_hook(hook_fn))  # type: ignore

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()

    def create_sae_pre_hook(self, sae, output, should_patch_activations):
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
        for module in model.saes.values():
            assert isinstance(module, SparseAutoencoder)
            module.load(Path(dir), device=device)

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
                assert isinstance(module, SparseAutoencoder)
                module.save(Path(dir))
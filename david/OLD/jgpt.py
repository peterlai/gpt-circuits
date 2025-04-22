import os
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Type, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_model, load_model

from models.mlpgpt import MLP_GPT
from models.mlpsparsified import MLPSparsifiedGPT
from models.sparsified import SparsifiedGPTOutput
from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import SparseAutoencoder, EncoderOutput 
from models.sae.jsae import JSAE

class JSparsifiedGPT(MLPSparsifiedGPT):
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
        assert len(config.n_features) == self.gpt.config.n_layer
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        
        self.saes = nn.ModuleDict(
            dict([(f'{x}', JSAE(config, loss_coefficients, self.gpt.transformer.h[x].mlp)) 
                  for x in self.layer_idxs])
        )
       
        # While a nice idea, it might break other code as TopKSAE
        # has a different save format 
        # for sae_key, sae in self.saes.items():
        #     self.saes[sae_key].sae_key = sae_key
        
        # def sae_save(self, dirpath: Path):
        #     """
        #     Save the sparse autoencoder to a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     save_model(self, str(weights_path))

        # def sae_load(self, dirpath: Path, device: torch.device):
        #     """
        #     Load the sparse autoencoder from a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     load_model(self, weights_path, device=device.type)
        
        # for sae_key, sae in self.saes.items():
        #     sae.save = lambda dirpath, current_sae=sae: sae_save(current_sae, dirpath)
        #     sae.load = lambda dirpath, device, current_sae=sae: sae_load(current_sae, dirpath, device)
        
        
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
        # print(cross_entropy_loss) # Optional: Keep for debugging
        # print(self.resid_mid_cache) # Optional: Keep for debugging
        #torch.cuda.synchronize()
        #print("SLOW DOWN BUDDY")
        
        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE layer
            ce_loss_increases = []
            for key in self.saes.keys():
                layer_idx, hook_loc = self.split_sae_key(key)
                recon_act = encoder_outputs[key].reconstructed_activations
                resid_mid = activations[f'{layer_idx}_residmid']
                
                sae_logits = self.gpt.forward_with_patched_activations(recon_act, resid_mid, layer_idx, hook_loc)
                sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_saes(activations_to_patch=self.saes.keys()):
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
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        activations: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            
            # run post hook for mlp to capture both inputs and outputs
            #seems to work even without disabling compiler
            @torch.compiler.disable(recursive=False)
            def mlp_hook_fn(module, inputs, outputs, layer_idx=layer_idx):
                # TODO: Why is inputs wrapped in a tuple, but outputs is not?
                # Why don't
                activations[f'{layer_idx}_mlpin'] = inputs[0]
                activations[f'{layer_idx}_mlpout'] = outputs
            
            # run pre hook for ln2 to capture resid_mid
            # need to sneak them out of the hook_fn
            
            # If you don't disable compiler, you get an error
            # about 0_residmid not being found???
            @torch.compiler.disable(recursive=False)
            def ln2_hook_fn(module, inputs, layer_idx=layer_idx):
                activations[f'{layer_idx}_residmid'] = inputs[0]

            
            hooks.append(ln2.register_forward_pre_hook(ln2_hook_fn))  # type: ignore
            hooks.append(mlp.register_forward_hook(mlp_hook_fn))  # type: ignore

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
 
    def create_sae_hook(self, sae, encoder_outputs, sae_key, should_patch_activations):
        """
        Create a forward pre-hook for the given layer index for applying sparse autoencoding.

        :param sae: SAE module to use for the forward pass.
        :param output: Encoder output to be updated.
        :param should_patch_activations: Whether to patch activations.
        """
        @torch.compiler.disable(recursive=False)  # type: ignore
        def sae_prehook_fn(module, inputs, outputs=None):
            """
            NOTE: Compiling seems to struggle with branching logic, and so we disable it (non-recursively).
            """
            hook_loc = sae_key.split('_')[-1]
            assert isinstance(inputs, tuple), f"inputs: {inputs}"
            if hook_loc == 'mlpin':
                assert outputs is None, f"outputs: {outputs}"
            if hook_loc == 'mlpout':
                assert outputs is not None, f"outputs: {outputs}"
            
            # TODO: Why is inputs wrapped in a tuple, but outputs is not?
            x = inputs[0] if hook_loc == 'mlpin' else outputs
            encoder_outputs[sae_key] = sae(x)

            # Patch activations if needed
            if should_patch_activations:
                return encoder_outputs[sae_key].reconstructed_activations
            else:
                return None

        return sae_prehook_fn
            
    def split_sae_key(self, sae_key: str) -> tuple[int, str]:
        """
        Split a SAE key into a layer index and hook location.
        """
        return int(sae_key.split('_')[0]), sae_key.split('_')[-1]
            
    @contextmanager
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        """
        # Dictionary for storing results
        encoder_outputs: dict[str, EncoderOutput] = {}

        # Register hooks
        hooks = []
        for sae_key in self.saes.keys():
            layer_idx, hook_loc = self.split_sae_key(sae_key)
            target = self.gpt.transformer.h[layer_idx].mlp
            sae = self.saes[sae_key]
            should_patch_activations = sae_key in activations_to_patch
            hook_fn = self.create_sae_hook(sae, encoder_outputs, sae_key, should_patch_activations)
            
            if hook_loc == 'mlpin':
                hooks.append(target.register_forward_pre_hook(hook_fn))  # type: ignore
            else:
                hooks.append(target.register_forward_hook(hook_fn))  # type: ignore

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()

    # function basically the same as SparsifiedGPT.load
    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load GPT model
        gpt = JSparsifiedGPT.load(dir, device=device)

        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)
        config.gpt_config = gpt.config

        # Create model using saved config
        model = MLPSparsifiedGPT(config, loss_coefficients, trainable_layers)
        model.gpt = gpt

        # Load SAE weights
        for module in model.saes.values():
            assert isinstance(module, SparseAutoencoder)
            module.load(Path(dir), device=device)

        return model

    # function basically the same as SparsifiedGPT.load_gpt_weights
    def load_gpt_weights(self, dir):
        """
        Load just the GPT model weights without loading SAE weights.
        """
        device = next(self.gpt.lm_head.parameters()).device
        self.gpt = JSparsifiedGPT.load(dir, device=device)

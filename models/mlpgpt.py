from models.gpt import GPT
from config.gpt.models import GPTConfig
import torch
from torch import Tensor
from jaxtyping import Float
import os
import json
from safetensors.torch import load_model

class MLP_GPT(GPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        
    @classmethod
    def load(cls, dir, device: torch.device):
        meta_path = os.path.join(dir, "model.json")
        weights_path = os.path.join(dir, "model.safetensors")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model = MLP_GPT(GPTConfig(**meta))

        load_model(model, weights_path, device=device.type)
        return model
        
    def forward_with_patched_activations(self, 
                                         patched_activations: Float[Tensor, "B T n_embd"], 
                                         resid_mid: Float[Tensor, "B T n_embd"],
                                         layer_idx: int,
                                         hook_loc: str) -> torch.Tensor:
        """
        Forward pass of the model with patched activations.
        0 : h[0].mlp_in
        1 : h[0].mlp_out
        2 : h[1].mlp_in
        3 : h[1].mlp_out
        ...
        :param resid_mid: Residual stream activations at the middle of the transformer block. Shape: (B, T, n_embd)
        :param patched_activations: Input activations. Shape: (B, T, n_embd)
        :param layer_idx: Layer index. 0 patches activations just before the first transformer block.
        :param mlp_idx: MLP index. 0 patches activations just before the first transformer block.
        """
        assert isinstance(patched_activations, torch.Tensor), f"x: {patched_activations}"
        if hook_loc == 'mlpin':
            mlp_out = self.transformer.h[layer_idx].mlp(patched_activations)
        elif hook_loc == 'mlpout':
            mlp_out = patched_activations
        else:
            raise ValueError(f"fo: Invalid hook location: {hook_loc}")
        
        resid_post = mlp_out + resid_mid
        # forward through transformer blocks starting with the specified layer
        for block in self.transformer.h[layer_idx+1:]:
            resid_post = block(resid_post)

        # forward through the final layernorm and the classifier
        resid_post_normalized = self.transformer.ln_f(resid_post)
        logits = self.lm_head(resid_post_normalized)

        return logits

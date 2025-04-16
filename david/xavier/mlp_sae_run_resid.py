# %%
#% load_ext autoreload  # COMMENT OUT or REMOVE
#% autoreload 2         # COMMENT OUT or REMOVE
# %%
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Change current working directory to parent
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

import torch
from pathlib import Path
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import StaircaseTopKSAE
from models.gpt import GPT
from models.mlpsparsified import MLPSparsifiedGPT
from data.tokenizers import ASCIITokenizer
from david.convert_to_tl import convert_gpt_to_transformer_lens
from david.convert_to_tl import run_tests as run_tl_tests
from transformer_lens.hook_points import HookPoint

from typing import Optional, Literal


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients()

gpt_dir = Path("checkpoints/shakespeare_64x4")
gpt = GPT.load(gpt_dir, device=device)

sae_config =SAEConfig(
        gpt_config=gpt.config,
        n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
        sae_variant=SAEVariant.TOPK,
        top_k = (10, 10, 10, 10, 10, 10, 10, 10)
    )

gpt_mlp = MLPSparsifiedGPT.load("checkpoints/mlp-topk.shakespeare_64x4", 
                                 loss_coefficients,
                                 trainable_layers = None,
                                 device = device)
gpt_mlp.to(device)
#tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)    


# %%# %%
# For Xavier's code
def run_resid(resid_mid, model, layer_idx = None, use_saes : Optional[Literal['mlpin', 'mlpout', 'both']] = None):
    """
    Runs the residual stream through the MLP block in layer layer_idx, optionally with the pre and post SAE layers.
    """
    assert layer_idx is not None, "layer_idx must be provided"
    blocks = model.gpt.transformer.h
    saes = model.saes
    
    resid = blocks[layer_idx].ln_2(resid_mid)
    
    if use_saes in ['mlpin', 'both']:  
        resid = saes[f"{layer_idx}_mlpin"](resid).reconstructed_activations
    
    resid = blocks[layer_idx].mlp(resid)
    
    if use_saes in ['mlpout', 'both']:
        resid = saes[f"{layer_idx}_mlpout"](resid).reconstructed_activations
    
    resid = resid + resid_mid
    
    return resid

resid_mid = torch.randn((3,7,64), device=device)
resid_post = run_resid(resid_mid, gpt_mlp, layer_idx = 0, use_saes = 'both')
# %%













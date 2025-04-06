# %%
%load_ext autoreload
%autoreload 2
# %%
import os

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
from david.jsae.sparsified import JSAESparsifiedGPT
from data.tokenizers import ASCIITokenizer
from david.convert_to_tl import convert_gpt_to_transformer_lens


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients()

gpt_dir = Path("checkpoints/shakespeare_64x4")
gpt = GPT.load(gpt_dir, device=device)

sae_config =SAEConfig(
        gpt_config=gpt.config,
        n_features=tuple(64 * n for n in (8, 8,8,8,8,8,8,8)),
        sae_variant=SAEVariant.STANDARD,
    )

gpt_mlp = JSAESparsifiedGPT(sae_config, loss_coefficients)
gpt_mlp.to(device)
#tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)

# %%
tok = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.long, device=device)
out = gpt_mlp(tok, targets=tok, is_eval=False)
# %%
tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)

tl_logits, cache = tl_gpt_mlp.run_with_cache(tok)

torch.testing.assert_close(out.logits, tl_logits)


for layer_idx in range(len(tl_gpt_mlp.blocks)):
    
    mlp_in = out.activations[f"{layer_idx}_mlpin"]
    mlp_out = out.activations[f"{layer_idx}_mlpout"]
    resid_mid = out.activations[f"{layer_idx}_residmid"]

    tl_mlp_in = cache[f'blocks.{layer_idx}.ln2.hook_normalized']
    tl_mlp_out = cache[f'blocks.{layer_idx}.hook_mlp_out']
    tl_resid_mid = cache[f'blocks.{layer_idx}.hook_resid_mid']
                            
    torch.testing.assert_close(mlp_in, tl_mlp_in)
    torch.testing.assert_close(mlp_out, tl_mlp_out)
    torch.testing.assert_close(resid_mid, tl_resid_mid)
    
    mlp_in_hat = out.reconstructed_activations[f"{layer_idx}_mlpin"]
    mlp_out_hat = out.reconstructed_activations[f"{layer_idx}_mlpout"]
    
    tl_mlp_in = cache[f'blocks.{layer_idx}.ln2.hook_normalized']
    tl_mlp_in_hat = gpt_mlp.saes[f"{layer_idx}_mlpin"](tl_mlp_in).reconstructed_activations
    
    tl_mlp_out = cache[f'blocks.{layer_idx}.hook_mlp_out']
    tl_mlp_out_hat = gpt_mlp.saes[f"{layer_idx}_mlpout"](tl_mlp_out).reconstructed_activations
    
    torch.testing.assert_close(mlp_in_hat, tl_mlp_in_hat)
    torch.testing.assert_close(mlp_out_hat, tl_mlp_out_hat)
    
    tl_mlp_in_featmag = gpt_mlp.saes[f"{layer_idx}_mlpin"].encode(tl_mlp_in)
    tl_mlp_out_featmag = gpt_mlp.saes[f"{layer_idx}_mlpout"].encode(tl_mlp_out)
    
    mlp_in_featmag = out.feature_magnitudes[f"{layer_idx}_mlpin"]
    mlp_out_featmag = out.feature_magnitudes[f"{layer_idx}_mlpout"]
    
    torch.testing.assert_close(mlp_in_featmag, tl_mlp_in_featmag)
    torch.testing.assert_close(mlp_out_featmag, tl_mlp_out_featmag)

# %%

out = gpt_mlp(tok, targets=tok, is_eval=True)
tl_logits, cache = tl_gpt_mlp.run_with_cache(tok)

torch.testing.assert_close(out.logits, tl_logits)


for layer_idx in range(len(tl_gpt_mlp.blocks)):
    
    
    mlp_in_hat = out.reconstructed_activations[f"{layer_idx}_mlpin"]
    mlp_out_hat = out.reconstructed_activations[f"{layer_idx}_mlpout"]
    
    tl_mlp_in = cache[f'blocks.{layer_idx}.ln2.hook_normalized']
    tl_mlp_in_hat = gpt_mlp.saes[f"{layer_idx}_mlpin"](tl_mlp_in).reconstructed_activations
    
    tl_mlp_out = cache[f'blocks.{layer_idx}.hook_mlp_out']
    tl_mlp_out_hat = gpt_mlp.saes[f"{layer_idx}_mlpout"](tl_mlp_out).reconstructed_activations
    
    torch.testing.assert_close(mlp_in_hat, tl_mlp_in_hat)
    torch.testing.assert_close(mlp_out_hat, tl_mlp_out_hat)
    
    tl_mlp_in_featmag = gpt_mlp.saes[f"{layer_idx}_mlpin"].encode(tl_mlp_in)
    tl_mlp_out_featmag = gpt_mlp.saes[f"{layer_idx}_mlpout"].encode(tl_mlp_out)
    
    mlp_in_featmag = out.feature_magnitudes[f"{layer_idx}_mlpin"]
    mlp_out_featmag = out.feature_magnitudes[f"{layer_idx}_mlpout"]
    
    torch.testing.assert_close(mlp_in_featmag, tl_mlp_in_featmag)
    torch.testing.assert_close(mlp_out_featmag, tl_mlp_out_featmag)





    
    
    
    
    
    
    
    
    
    
    
    
    


# %%

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
    
# %% 
def generate_with_mlpsae(tokens, sparse_model, keys = None, max_length=50, temperature=0.7):
    if keys is None:
        keys = sparse_model.saes.keys() #use all layers
    
    for _ in range(max_length):
        
        with sparse_model.use_saes(activations_to_patch = keys):
            output = sparse_model(tokens)

        logits = output.logits[:, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokenizer.decode_sequence(tokens[0].tolist())


tokenizer = ASCIITokenizer()
tokens = tokenizer.encode("KING HENRY:")
device = next(gpt_mlp.parameters()).device
tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)

# How to use
#generated_sae = generate_with_mlpsae(tokens, gpt_mlp, tokenizer, keys = ['0_mlpin', '0_mlpout'])

for key in gpt_mlp.saes.keys():
    print(key)
    print("========")
    print(generate_with_mlpsae(tokens, gpt_mlp, keys = [key]))

print("ALL SAE LAYERS")
print('======')
print(generate_with_mlpsae(tokens, gpt_mlp))


# %%# %%
# For Xavier's code
def run_resid(resid, model, layer_idx):
    """
    Runs the residual stream through the MLP layer, with the pre and post SAE layers.
    """
    resid_norm = model.blocks[layer_idx].ln2(resid)
    resid_premlp = model.saes[f"{layer_idx}_mlpin"](resid_norm)
    resid_postmlp = model.blocks[layer_idx].mlp(resid_premlp)
    resid_post = model.saes[f"{layer_idx}_mlpout"](resid_postmlp)
    resid_post = resid_post + resid

    return resid_post









# %%
# %load_ext autoreload  # COMMENT OUT or REMOVE
# %autoreload 2         # COMMENT OUT or REMOVE
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
from david.jsae.jsparsified import JSAESparsifiedGPT
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

gpt_mlp = JSAESparsifiedGPT.load("checkpoints/mlp-topk.shakespeare_64x4", 
                                 loss_coefficients,
                                 trainable_layers = None,
                                 device = device)
gpt_mlp.to(device)
#tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)

# %%
prompt = "Second Servingman:\nI will not so"
tokenizer = ASCIITokenizer()
tokens = tokenizer.encode(prompt)
tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)

x = tokens[:, :-1].contiguous()
y = tokens[:, 1:].contiguous()
out = gpt_mlp(x, targets=y, is_eval=False)
# %%
tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)
run_tl_tests(gpt_mlp.gpt, tl_gpt_mlp, sae_config)

tl_logits, cache = tl_gpt_mlp.run_with_cache(x)

torch.testing.assert_close(out.logits, tl_logits, atol=3e-5, rtol=3e-5)


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
print("forward passed")
# %%

print("Testing forward_with_patched_activations")
resid_mid = torch.randn((2,3,64), device=device)
patched_activations = torch.randn((2,3,64), device=device)

for layer_idx in range(4): #start at layers 0,1,2,3
    for loc in ['mlpin', 'mlpout']:

        if loc == 'mlpin':
            mlp_out = tl_gpt_mlp.blocks[layer_idx].mlp(patched_activations)
        else:
            mlp_out = patched_activations
        
        resid_post = mlp_out + resid_mid
        tl_logits = tl_gpt_mlp(resid_post, start_at_layer = layer_idx + 1)
    
        logits = gpt_mlp.gpt.forward_with_patched_activations(patched_activations, resid_mid, layer_idx, loc)

        torch.testing.assert_close(tl_logits, logits)
print("forward_with_patched_activations passed")

# %%
print("Test logits end-to-end")

from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from torch.nn import functional as F
from transformer_lens.utils import lm_cross_entropy_loss
import einops
tl_hooks = []
for layer_idx in range(len(tl_gpt_mlp.blocks)):
    for loc in ['mlpin', 'mlpout']:
        sae_key = f"{layer_idx}_{loc}"
        
        
        def hook_fn(act : Float[Tensor, "B T n_embd"], 
                    hook: HookPoint,
                    sae_key = sae_key):
            reconstructed_act = gpt_mlp.saes[sae_key](act).reconstructed_activations
            return reconstructed_act
        
        if loc == 'mlpin':
            hook_name = f'blocks.{layer_idx}.ln2.hook_normalized'
        elif loc == 'mlpout':
            hook_name = f'blocks.{layer_idx}.hook_mlp_out'
        else:
            raise ValueError(f"Invalid location: {loc}")
            
        tl_hooks.append((hook_name, hook_fn))
        
# Note that transformer lens takes the entire sequence as input
# but Peters code expectes x, y = tokens[:, :-1], tokens[:, 1:] as input

tl_logits, tl_loss = tl_gpt_mlp.run_with_hooks(tokens, fwd_hooks = tl_hooks, return_type = 'both')
tl_logits = tl_logits[:, :-1, :]

with gpt_mlp.use_saes(activations_to_patch = gpt_mlp.saes.keys()):
    e2e_logits, e2e_loss = gpt_mlp.gpt(x, targets=y)
    
    
# %% 
def generate_sae(sparse_model, tokenizer, prompt, max_length=50, temperature=0.7) -> str:
    """
    Generate text from a prompt using the model
    """
    device = next(sparse_model.parameters()).device

    tokens = tokenizer.encode(prompt)
    tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)
    
        
    def generate_tokens(tokens, keys):
        for _ in range(max_length):
            
            with sparse_model.use_saes(activations_to_patch = keys):
                output = sparse_model(tokens)

            logits = output.logits[:, -1]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokenizer.decode_sequence(tokens[0].tolist())
    
    for key in sparse_model.saes.keys():
        print(key)
        print("========")
        print(generate_tokens(tokens, [key]))
    
    
    print("ALL SAE LAYERS")
    print('======')
    print(generate_tokens(tokens, sparse_model.saes.keys()))
    
tokenizer = ASCIITokenizer()
generated_sae = generate_sae(gpt_mlp, tokenizer, "KING HENRY:")
# %%
david_e2e_logits = einops.rearrange(e2e_logits, 'b t d -> (b t) d')
david_loss = F.cross_entropy(david_e2e_logits, y.flatten())
    
print(e2e_loss, tl_loss, david_loss)
torch.testing.assert_close(e2e_logits, tl_logits)
torch.testing.assert_close(david_loss, tl_loss)
torch.testing.assert_close(e2e_loss, tl_loss)
torch.testing.assert_close(david_loss, tl_loss)
print("logits end-to-end passed")
    
    

out = gpt_mlp(x,y,is_eval=True)
# %%

# %%


import os
import sys
import torch
# Add root directory to sys.path dynamically
# sys.path hack to run in VSCode interactive session
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# %%
from models.gpt import GPT
from models.sparsified import SparsifiedGPT
from safetensors.torch import load_model
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer
from models.sae.topk import TopKSAE
from config.sae.models import SAEConfig
from david.utils import generate
from config.sae.training import LossCoefficients
import json 
from pathlib import Path


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpt_dir = Path("checkpoints/shakespeare_64x4")
gpt = GPT.load(gpt_dir, device=device)
tokenizer = ASCIITokenizer()
print(generate(gpt, tokenizer, "Today I thought,", max_length=100))


sae_dir = Path("checkpoints/topk.shakespeare_64x4")
sae_config_dir = sae_dir / "sae.json"
with open(sae_config_dir, "r") as f:
    meta = json.load(f)
config = SAEConfig(**meta)
config.gpt_config = gpt.config

# %%
# Create model using saved config
print("Creating SparsifiedGPT model...")
model = SparsifiedGPT(config)
model.gpt = gpt

# Load SAE weights
print("Loading SAE weights...")
for layer_name, module in model.saes.items():
    weights_path = os.path.join(sae_dir, f"sae.{layer_name}.safetensors")
    load_model(module, weights_path, device=device.type)
    

model.to(device)

example_input = "Thou shall not give unto me"
tokens = tokenizer.encode(example_input)
input_ids = torch.tensor(tokens, device=device).unsqueeze(0).to(device)
sparse_gpt_output = model(input_ids[:, :-1], input_ids[:, 1:])

print("SparsifiedGPT Output:")
print(f"Cross Entropy Loss: {sparse_gpt_output.cross_entropy_loss}")
print(f"Cross Entropy Loss Increases: {sparse_gpt_output.ce_loss_increases}")
print(f"Compound Cross Entropy Loss Increase: {sparse_gpt_output.compound_ce_loss_increase}")
print(f"SAE Loss Components: {sparse_gpt_output.sae_loss_components}")


# for sae in saes:
#     sae.load()


# %%

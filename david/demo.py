# %%
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_grad_enabled(False)
# %%
from models.gpt import GPT
from models.sparsified import SparsifiedGPT
from safetensors.torch import load_model
from data.tokenizers import ASCIITokenizer
from config.sae.models import SAEConfig
from david.utils import generate
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_gpt_model():
    gpt_dir = Path("checkpoints/shakespeare_64x4")
    gpt = GPT.load(gpt_dir, device=device)
    tokenizer = ASCIITokenizer()
    return gpt, tokenizer

def load_sae_config(gpt):
    sae_dir = Path("checkpoints/topk.shakespeare_64x4")
    sae_config_dir = sae_dir / "sae.json"
    with open(sae_config_dir, "r") as f:
        meta = json.load(f)
        print(f"SAE config: {meta}")
    config = SAEConfig(**meta)
    config.gpt_config = gpt.config
    return config, sae_dir

def create_sparsified_gpt_model(config, gpt, sae_dir, device):
    model = SparsifiedGPT(config)
    model.gpt = gpt

    for layer_name, module in model.saes.items():
        weights_path = os.path.join(sae_dir, f"sae.{layer_name}.safetensors")
        load_model(module, weights_path, device=device.type)

    model.to(device)
    return model

def load_and_prepare_data_loader(device):
    val_activations_path = "data/shakespeare/val_000000.npy"
    val_activations = np.load(val_activations_path, allow_pickle=False).astype(np.int32)
    val_activations = torch.tensor(val_activations, device=device)
    
    chunk_size = 128
    batch_size = 32
    N = val_activations.shape[0]
    val_activations = val_activations[:(N//chunk_size)*chunk_size].long()
    val_activations = val_activations.view(-1, chunk_size)
    val_loader = torch.utils.data.DataLoader(val_activations, batch_size=batch_size, shuffle=False)
    
    return val_loader

def evaluate_model(model, val_loader):
    ce_losses = []
    ce_increases = []
    compound_ce_loss_increases = []
    for batch in tqdm(val_loader, desc="Evaluating model"):
        input_ids = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        sparse_gpt_output = model(input_ids, targets, is_eval=True)

        ce_losses.append(sparse_gpt_output.cross_entropy_loss.item())
        ce_increases.append(sparse_gpt_output.ce_loss_increases.cpu())
        compound_ce_loss_increases.append(sparse_gpt_output.compound_ce_loss_increase)

    ce_increases = torch.stack(ce_increases, dim=0).mean(dim=0)
    ce_losses = torch.tensor(ce_losses).mean()
    compound_ce_loss_increases = torch.stack(compound_ce_loss_increases, dim=0).mean()

    print(f"Performance on Validation Set:")
    print(f"Loss: {ce_losses:.4f}")
    print(f"Loss Inc: {ce_increases}")
    print(f"Compound Loss Inc: {compound_ce_loss_increases:.4f}")

def main():
    gpt, tokenizer = load_gpt_model()
    print("-----------------")
    print(generate(gpt, tokenizer, "Today I thought,", max_length=100))
    print("-----------------")
    config, sae_dir = load_sae_config(gpt)
    model = create_sparsified_gpt_model(config, gpt, sae_dir, device)
    val_loader = load_and_prepare_data_loader(device)
    evaluate_model(model, val_loader)
# %%
if __name__ == "__main__":
    main()
# %%
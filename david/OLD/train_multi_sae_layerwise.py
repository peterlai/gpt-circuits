import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from config.gpt.models import gpt_options
from config.gpt.training import options as gpt_train_options
from models.gpt import GPT
from models.sae.multilayer import MultiLayerSAEBase
from config.sae.training import SAETrainingConfig, LossCoefficients
from config.sae.models import SAEConfig, SAEVariant
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from david.cache_activations import cache_activations

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'shakespeare_64x4'

# Define paths
checkpoints = "checkpoints"
data_root = "data"
model_path = os.path.join(checkpoints, model_name)
dataset_path = os.path.join(data_root, model_name)

# Load model configuration
config = gpt_train_options[model_name]
model = GPT(config.gpt_config)
model = model.load(model_path, device=config.device)
model.to(device)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    cache_activations(dataset_path, model_path, batch_size=64, device=device)

cache_train = np.load(os.path.join(dataset_path, "train_activations.npy"), allow_pickle=False)
cache_val = np.load(os.path.join(dataset_path, "val_activations.npy"), allow_pickle=False)

# Shared training parameters
shakespeare_64x4_defaults = {
    "data_dir": "data/shakespeare",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "warmup_steps": 750,
    "max_steps": 7500,
    "decay_lr": True,
    "min_lr": 1e-4,
}

starcasex8_options = SAEConfig(
    name="staircasex8.shakespeare_64x4",
    gpt_config=gpt_options["ascii_64x4"],
    n_features=64 * 8,
    sae_variant=SAEVariant.STANDARD,
)

sae_train_config = SAETrainingConfig(
    name="multilayer.shakespeare_64x4",
    sae_config=starcasex8_options,
    **shakespeare_64x4_defaults,
    loss_coefficients=LossCoefficients(
        sparsity=(0.06, 0.24, 0.8, 1.0, 2.5),
    )
)

sae = MultiLayerSAEBase(sae_train_config.sae_config, sae_train_config.loss_coefficients)
sae.to(device)

train_data = cache_train
val_data = cache_val

train_tensor = torch.from_numpy(train_data).to(device)
val_tensor = torch.from_numpy(val_data).to(device)

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=shakespeare_64x4_defaults['batch_size'],
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=shakespeare_64x4_defaults['batch_size'],
    shuffle=False,
)

optimizer = torch.optim.AdamW(sae.parameters(), lr=shakespeare_64x4_defaults['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# Training loop
for layer_idx in range(sae.n_layers):
    print(f"Training layer {layer_idx}")
    sae.train()
    for epoch in range(1):  # Train each layer for one epoch
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            output = sae(batch_data, layer_idx=layer_idx)
            loss = output.loss.total
            loss.backward()
            optimizer.step()

    # Copy weights to the next layer and freeze current layer's weights
    if layer_idx < sae.n_layers - 1:
        with torch.no_grad():
            next_layer_start = (layer_idx + 1) * sae.feature_size
            sae.W_enc.data[:, next_layer_start:next_layer_start + sae.feature_size] = sae.W_enc.data[:, layer_idx * sae.feature_size:(layer_idx + 1) * sae.feature_size]
            sae.b_enc.data[next_layer_start:next_layer_start + sae.feature_size] = sae.b_enc.data[layer_idx * sae.feature_size:(layer_idx + 1) * sae.feature_size]

        # Freeze current layer's weights
        for param in [sae.W_enc, sae.b_enc]:
            param.requires_grad = False

    # Evaluate after each layer training
    sae.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_data = val_batch[0].to(device)
            val_output = sae(val_data, layer_idx=layer_idx)
            val_losses.append(val_output.loss.total.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Layer {layer_idx} - Validation Loss: {avg_val_loss:.4f}")

# Save the final model
torch.save(sae.state_dict(), os.path.join(checkpoints, f"{model_name}_multi_sae_layerwise.pt")) 
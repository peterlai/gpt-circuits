import os
import torch
import numpy as np
from tqdm import tqdm
from config.gpt.models import gpt_options
from config.sae.training import options as sae_train_options
from models.gpt import GPT
from models.sae.topk import TopKSAE
from torch.utils.data import TensorDataset, DataLoader
from config.sae.training import SAETrainingConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'topk.shakespeare_64x4'

# Define paths
checkpoints = "checkpoints"
data_root = "data"
model_path = os.path.join(checkpoints, model_name)
dataset_path = os.path.join(data_root, model_name)

# Load model configuration
config: SAETrainingConfig = sae_train_options[model_name]
gpt_config = config.sae_config.gpt_config
model = GPT(gpt_config)
model = model.load(model_path, device=device)
model.to(device)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    # Assuming cache_activations is a function to cache activations
    cache_activations(dataset_path, model_path, batch_size=64, device=device)

cache_train = np.load(os.path.join(dataset_path, "train_activations.npy"), allow_pickle=False)
cache_val = np.load(os.path.join(dataset_path, "val_activations.npy"), allow_pickle=False)

# Create datasets and dataloaders
train_tensor = torch.from_numpy(cache_train).to(device)
val_tensor = torch.from_numpy(cache_val).to(device)

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
)

# Initialize the TopKSAE model
sae = TopKSAE(
    layer_idx=0,  # Start with the first layer
    config=config.sae_config,
    loss_coefficients=config.loss_coefficients
)
sae.to(device)

optimizer = torch.optim.AdamW(sae.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=config.min_lr)

# Training loop
sae.train()
best_val_loss = float('inf')
step = 0
optimizer.zero_grad()

progress_bar = tqdm(total=config.max_steps, desc='Training')

while step < config.max_steps:
    for batch in train_loader:
        if step >= config.max_steps:
            break

        # Move data to device
        batch_data = batch[0].to(device)

        # Forward pass
        output = sae(batch_data)
        loss = output.loss.total

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate with scheduler
        if config.decay_lr:
            scheduler.step()

        # Evaluation
        if step % config.eval_interval == 0:
            sae.eval()
            val_losses = []

            with torch.no_grad():
                for val_batch in val_loader:
                    val_data = val_batch[0].to(device)
                    val_output = sae(val_data)
                    val_losses.append(val_output.loss.total.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'step': step,
                    'sae_state_dict': sae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(checkpoints, f"{model_name}_topk_sae_best.pt"))

            progress_bar.set_postfix({
                'lr': optimizer.param_groups[0]['lr'],
                'val_loss': f'{avg_val_loss:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}',
            })

            sae.train()

        step += 1
        progress_bar.update(1)

progress_bar.close() 
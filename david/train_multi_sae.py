# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
import torch
import numpy as np
from tqdm import tqdm
from config.gpt.training import options as gpt_options
from models.gpt import GPT
from models.sae.multilayer import MultiLayerSAEBase
import warnings
import argparse

from jaxtyping import Float
from torch import Tensor
from config.sae.training import options as sae_options
from torch.utils.data import TensorDataset, DataLoader
from cache_activations import cache_activations
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'shakespeare_64x4'
# Add root directory to sys.path dynamically


# Define paths
checkpoints = "../checkpoints"
data_root = "../data"
model_path = os.path.join(checkpoints, model_name)
dataset_path = os.path.join(data_root, model_name)

# Load model configuration
config = gpt_options[model_name]
model = GPT(config.gpt_config)
model = model.load(model_path, device=config.device)
model.to(device)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    cache_activations(dataset_path, model_path, batch_size=64, device=device)

cache_train = np.load(os.path.join(dataset_path, "train_activations.npy"), allow_pickle=False)
cache_val = np.load(os.path.join(dataset_path, "val_activations.npy"), allow_pickle=False)
# %%

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

sae_train_config = sae_options["staircase.shakespeare_64x4"]
sae = MultiLayerSAEBase(sae_train_config.sae_config, sae_train_config.loss_coefficients)
sae = sae.to(device)
# %%

# def train_multi_sae(
#     model: MultiLayerSAEBase,
#     train_data: np.ndarray,
#     val_data: np.ndarray,
#     config: dict = shakespeare_64x4_defaults,
# ):
#     """Train a multi-layer sparse autoencoder using DataLoader"""


# trained_model = train_multi_sae(
#     model=multi_layer_sae,
#     train_data=cache_train,
#     val_data=cache_val
# )

config = shakespeare_64x4_defaults

train_data = cache_train
val_data = cache_val

# Create datasets and dataloaders
train_tensor = torch.from_numpy(train_data).to(device)
val_tensor = torch.from_numpy(val_data).to(device)

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'],
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
)

# Setup optimizer

class ScaledAdamW(torch.optim.AdamW):
    def __init__(self, params, sae, lr=0.001, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.sae = sae
    
    def step(self):
        # Scale gradients
        n_layers = self.sae.n_layers
        feature_size = self.sae.feature_size
        
        with torch.no_grad():
            param_groups = [
                (self.sae.W_dec.grad, lambda idx: (idx*feature_size, (idx+1)*feature_size), lambda slice_idx: (slice(*slice_idx), slice(None))),
                (self.sae.W_enc.grad, lambda idx: (idx*feature_size, (idx+1)*feature_size), lambda slice_idx: (slice(None), slice(*slice_idx))),
                (self.sae.b_enc.grad, lambda idx: (idx*feature_size, (idx+1)*feature_size), lambda slice_idx: (slice(*slice_idx),))
            ]
            
            for grad, slice_fn, index_fn in param_groups:
                if grad is not None:
                    for layer_idx in range(n_layers):
                        slice_idx = slice_fn(layer_idx)
                        scaling_factor = 1.0 / (n_layers - layer_idx)
                        idx = index_fn(slice_idx)
                        grad[idx] *= scaling_factor
        
        # Call parent step
        super().step()

# Usage
optimizer = ScaledAdamW(sae.parameters(), sae, lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)


# Learning rate scheduler
if config['decay_lr']:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['max_steps'] - config['warmup_steps'],
        eta_min=config['min_lr']
    )
# %%
# Training loop
sae.train()
best_val_loss = float('inf')
step = 0
optimizer.zero_grad()  # Initialize gradients

progress_bar = tqdm(total=config['max_steps'], desc='Training')

# Learning rate warmup
def get_lr(step, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr

while step < config['max_steps']:
    for batch in train_loader:
        if step >= config['max_steps']:
            break
            
        # Move data to device
        batch_data = batch[0].to(device)
        
        # Accumulate loss across all layers for this batch
        total_batch_loss = 0
        
        # Process each layer in the SAE sequentially
        for layer_idx in range(sae.n_layers):
            # Forward pass through this layer
            output = sae(batch_data, layer_idx=layer_idx)
            layer_loss = output.loss.total
            
            total_batch_loss += layer_loss
        
        # Now backward pass on accumulated loss with gradient accumulation
        total_batch_loss = total_batch_loss / config['gradient_accumulation_steps']
        total_batch_loss.backward()
        
        # Gradient accumulation and optimization
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            # Learning rate warmup
            if step < config['warmup_steps']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(
                        step, 
                        config['warmup_steps'], 
                        config['learning_rate'], 
                        config['min_lr']
                    )
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate with scheduler after warmup
            if config['decay_lr'] and step >= config['warmup_steps']:
                scheduler.step()
        
                # Evaluation
        if step % config['eval_interval'] == 0:
            sae.eval()
            val_losses = []
            
            # Initialize metrics to track per layer
            layer_metrics = {i: {'l0': 0.0, 'reconstruct': 0.0, 'sparsity': 0.0} for i in range(sae.n_layers)}
            n_batches = 0
            
            with torch.no_grad():
                # Evaluate all layers for each validation batch
                for val_batch in val_loader:
                    val_data = val_batch[0].to(device)
                    n_batches += 1
                    
                    # Accumulate validation loss across all layers
                    batch_val_loss = 0
                    for layer_idx in range(sae.n_layers):
                        val_output = sae(val_data, layer_idx=layer_idx)
                        batch_val_loss += val_output.loss.total.item() / sae.n_layers
                        
                        # Accumulate metrics for this layer
                        layer_metrics[layer_idx]['l0'] += val_output.loss.l0.item()
                        layer_metrics[layer_idx]['reconstruct'] += val_output.loss.reconstruct.item()
                        layer_metrics[layer_idx]['sparsity'] += val_output.loss.sparsity.item()
                    
                    val_losses.append(batch_val_loss)
                    
                    if len(val_losses) >= config['eval_steps']:
                        break
                
                # Calculate average metrics per layer
                for layer_idx in range(sae.n_layers):
                    for metric in layer_metrics[layer_idx]:
                        layer_metrics[layer_idx][metric] /= n_batches
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                
                # Print layer-wise metrics
                print(f"\nStep {step} - Layer-wise metrics:")
                for layer_idx in range(sae.n_layers):
                    print(f"  Layer {layer_idx}: L0={layer_metrics[layer_idx]['l0']:.2f}, "
                        f"Recon={layer_metrics[layer_idx]['reconstruct']:.4f}, "
                        f"Sparsity={layer_metrics[layer_idx]['sparsity']:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'step': step,
                        'sae_state_dict': sae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'layer_metrics': layer_metrics,  # Save metrics in checkpoint
                    }, os.path.join(checkpoints, f"{model_name}_multi_sae_best.pt"))
                
                progress_bar.set_postfix({
                    'lr': optimizer.param_groups[0]['lr'],
                    'val_loss': f'{avg_val_loss:.4f}',
                    'best_val_loss': f'{best_val_loss:.4f}',
                    'avg_l0': f"{sum(m['l0'] for m in layer_metrics.values())/sae.n_layers:.2f}"
                })
            
            sae.train()
        
        step += 1
        progress_bar.update(1)
        
        if step >= config['max_steps']:
            break

progress_bar.close()
#return model

    
# %%
from safetensors.torch import save_file
import torch
import os
import json
def save_sae_to_huggingface(sae: MultiLayerSAEBase, save_dir: str, model_name: str = "sae"):
    """
    Save the MultiLayerSAEBase model to Hugging Face format using safetensors.
    
    Args:
        sae: The MultiLayerSAEBase instance to save
        save_dir: Directory where the model will be saved
        model_name: Name of the model file (default: "multi_sae")
    """
    # Create save directory if it doesn’t exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare the state dictionary with contiguous tensors
    state_dict = {
        "W_dec": sae.W_dec.contiguous(),
        "W_enc": sae.W_enc.contiguous(),
        "b_enc": sae.b_enc.contiguous(),
        "b_dec": sae.b_dec.contiguous(),
    }
    
    # Save the tensors using safetensors
    save_path = os.path.join(save_dir, f"{model_name}.safetensors")
    save_file(state_dict, save_path)
    
    # Prepare configuration dictionary, converting device to string
    gpt_config_dict = vars(sae.config.gpt_config)  # Convert dataclass to dict
    if "device" in gpt_config_dict and isinstance(gpt_config_dict["device"], torch.device):
        gpt_config_dict["device"] = str(gpt_config_dict["device"])  # Convert torch.device to string
    
    config_dict = {
        "feature_size": sae.feature_size,
        "n_layers": sae.n_layers,
        "gpt_config": gpt_config_dict,
        "l1_coefficient": sae.l1_coefficient if sae.l1_coefficient is not None else None,
    }
    
    # Save configuration as JSON
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Model saved to {save_path}")

# Example usage
sae = MultiLayerSAEBase(config=sae_train_config.sae_config, loss_coefficients=sae_train_config.loss_coefficients)
save_dir = "../checkpoints/multi-layer.shakespeare_64x4"
save_sae_to_huggingface(sae, save_dir)
# %%

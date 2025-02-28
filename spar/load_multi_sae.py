# %%
from safetensors.torch import load_file
import torch
import json
from models.sae.multilayer import MultiLayerSAEBase
from config.sae.training import SAEConfig, LossCoefficients  # Adjust based on your actual config classes
import os
def load_sae_from_huggingface(save_dir: str, model_name: str = "multi_sae", device: str = "cuda"):
    """
    Load a MultiLayerSAEBase model from Hugging Face format using safetensors.
    
    Args:
        save_dir: Directory where the model is saved
        model_name: Name of the model file (default: "multi_sae")
        device: Device to load the model onto (default: "cuda")
    
    Returns:
        MultiLayerSAEBase: Loaded model instance
    """
    # Load configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Reconstruct gpt_config, converting device string back to torch.device if needed
    gpt_config_dict = config_dict["gpt_config"]
    if "device" in gpt_config_dict:
        gpt_config_dict["device"] = torch.device(gpt_config_dict["device"])  # Convert string back to torch.device
    
    # Reconstruct SAEConfig (adjust based on your actual SAEConfig class)
    gpt_config = type(sae_train_config.sae_config.gpt_config)(**gpt_config_dict)  # Assuming a dataclass
    sae_config = SAEConfig(
        gpt_config=gpt_config,
        n_features=config_dict["feature_size"],
        # Add other required fields if necessary
    )
    
    # Reconstruct LossCoefficients if provided
    loss_coefficients = LossCoefficients(sparsity=config_dict["l1_coefficient"]) if config_dict["l1_coefficient"] else None
    
    # Initialize the model
    sae = MultiLayerSAEBase(config=sae_config, loss_coefficients=loss_coefficients)
    
    # Load the state dictionary
    model_path = os.path.join(save_dir, f"{model_name}.safetensors")
    state_dict = load_file(model_path)
    
    # Load tensors into the model
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path}")
    return sae

# Example usage
save_dir = "../checkpoints/multi-layer.shakespeare_64x4"
loaded_sae = load_sae_from_huggingface(save_dir, model_name="sae", device="cuda")
# %%

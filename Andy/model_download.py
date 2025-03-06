from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Define model repo and local folder
model_name = "davidquarel/standard.shakespeare_64x4"
local_dir = "checkpoints"

# Download the model and tokenizer
snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

print(f"Model downloaded successfully into {local_dir}")
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Define model repo and local folder
model_name = "davidquarel/jsae.shakespeare_64x4-sparsity-1.0e-04"
local_dir = "checkpoints/jsae.shakespeare_64x4-sparsity-1.0e-04"

sparsities = ['1.0e-04', '1.0e-05', '1.0e-06', '1.0e-07', '1.0e-08', '1.0ep00', '1.0ep01', 
'3.3e-02', '3.3e-04', '3.3e-05', '3.3e-06', '3.3e-07', '3.3e-08', '3.3ep00', '3.3e-01', '0.0ep00']

model_names = ['davidquarel/jsae.shakespeare_64x4-sparsity-' +sparsity for sparsity in sparsities]
local_dirs = ['checkpoints/jsae.shakespeare_64x4-sparsity-' +sparsity for sparsity in sparsities]

with open('commands.txt', 'w') as f:
    
    for model_name, local_dir in zip(model_names, local_dirs):
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

        print(f"Model downloaded successfully into {local_dir}")
        bash_command = f'python Andy/compute_attributions.py --config=jsae.topkx8.shakespeare_64x4 --load_from={local_dir} --save_to=Andy/data --data_dir=data/shakespeare --num_batches=24 --batch_size=24 --attribution_method=ig'
        f.write(bash_command + '\n')    
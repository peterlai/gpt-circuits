from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub import upload_file



sparsities = ['1.0e-03', '1.2e-03','1.5e-03', '1.8e-03', '2.2e-03', '2.7e-03', '3.3e-03', '3.9e-03', '4.7e-03', '5.6e-03', '6.8e-03', '1.0e-02']
#sparsities = ['1.0e-04', '1.0e-05', '1.0e-06', '1.0e-07', '1.0e-08', '1.0ep00', '1.0ep01', 
#'3.3e-02', '3.3e-04', '3.3e-05', '3.3e-06', '3.3e-07', '3.3e-08', '3.3ep00', '3.3e-01', '0.0ep00']

model_names = ['davidquarel/jsae.shk_64x4-sparse-' +sparsity+'-steps-20k' for sparsity in sparsities]
local_dirs = ['checkpoints/jsae.shakespeare_64x4-sparsity-' +sparsity+'-steps-20k' for sparsity in sparsities]

with open('commands.txt', 'w') as f:
    
    for idx, model_name in enumerate(model_names):
        
        local_dir = local_dirs[idx]
        sparsity = sparsities[idx]
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"Model downloaded successfully into {local_dir}")
        bash_command = f'python Andy/compute_attributions.py --config=jsae.topkx8.shakespeare_64x4 --load_from={local_dir} --save_to=Andy/data --data_dir=data/shakespeare --save_name=jsae.shakespeare_64x4-sparsity-{sparsity}-20k --num_batches=24 --batch_size=24 --attribution_method=ig'
        f.write(bash_command + '\n')
        
        #Sloppily using same code to load it
        #repo_id = "algo2217/SPAR-attributions"
        #path = "Andy/data/jsae.shakespeare_64x4-sparsity-" + sparsity + "-20k.safetensors"

        #upload_file(
            #path_or_fileobj=path,
            #path_in_repo="jsae.shakespeare_64x4-sparsity-" + sparsity + "-20k.safetensors",
            #repo_id=repo_id,
            #repo_type="dataset"
        )
       # print(f"uploaded file at {path}")

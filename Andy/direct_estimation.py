import sys
sys.path.append('/Users/andrewgordon/Documents/ML/TeamStefan/gpt-circuits/')


import torch as t
import torch.nn as nn
from torch import Tensor
from torch.autograd.functional import jacobian
import os

from safetensors.torch import save_file

import einops

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from config.gpt.training import options
from config.sae.models import sae_options



from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from models.sae import SparseAutoencoder
from typing import Callable

from data.dataloaders import TrainingDataLoader
TensorFunction = Callable[[Tensor], Tensor]

def direct_estimate(model: SparsifiedGPT, layer0: int, layer1: int, ds: TrainingDataLoader, nbatches: int=32):
    with t.no_grad():
        epsilon = .001
        
        assert layer0 < layer1
        assert layer0 >= 0
        assert layer1 <= model.gpt.config.n_layer
        sae0 = model.saes[f'{layer0}']
        sae1 = model.saes[f'{layer1}']
        

        #define function that goes from feature magnitudes in layer0 to feature magnitudes in layer1
        #Q: Is this good form? I need it to make my Sequential object below
        class Sae0Decode(nn.Module):
            def forward(self, x):
                return sae0.decode(x)
            
        class Sae1Encode(nn.Module):
            def forward(self, x):
                return sae1.encode(x)

        #construct function from Sae0 to Sae1
        forward_list = [Sae0Decode()] + [model.gpt.transformer.h[i] for i in range(layer0, layer1)] + [Sae1Encode()]
        forward = t.nn.Sequential(*forward_list)

        source_size, _ = sae0.W_dec.shape
        target_size, _ = sae1.W_dec.shape

        scores = t.zeros((source_size, target_size), device = model.gpt.config.device)
        occurences = t.zeros((model.gpt.config.block_size, source_size), device = model.gpt.config.device)


        for _ in range(nbatches):
            input, _ = ds.next_batch(model.gpt.config.device) #get batch of inputs 
            output = model.forward(input.long(), targets=None, is_eval=True)
            feature_magnitudes0 = output.feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)
            feature_magnitudes1 = output.feature_magnitudes[layer1]
            batchsize = feature_magnitudes0.shape[0]

            batch_patches = []
            batch_indices = []

            for b in range(batchsize):
                up = feature_magnitudes0[b:b+1].contiguous()  # (1, seqlen, source_size)
                down = feature_magnitudes1[b:b+1]  # (1, seqlen, target_size)
                nz = (up > epsilon).nonzero(as_tuple=True)  # Get indices as tuples
                for idx in zip(*nz):
                    patch = up.clone()  # Clone once per batch element
                    patch[0, idx[1], idx[2]] = 0  # Modify in-place
                    batch_patches.append(patch)
                    batch_indices.append((b, idx[1], idx[2]))

        # Run a single forward pass for all patches
        if batch_patches:
            batch_patches = t.cat(batch_patches, dim=0)  # Concatenate patches into a single tensor
            results = forward(batch_patches)  # Forward pass for all patches

            # Update scores and occurrences
            for i, (b, seq_idx, src_idx) in enumerate(batch_indices):
                scores[seq_idx, src_idx] += (results[i] - feature_magnitudes1[b]).abs().sum(dim=(0, 1))
                occurences[seq_idx, src_idx] += 1
        return scores, occurences

        """         mask = t.ones((1, model.gpt.config.block_size, source_size), device = model.gpt.config.device) #mask for the input to the forward function
        for _ in range(nbatches):

            input, _ = ds.next_batch(model.gpt.config.device) #get batch of inputs 
            output = model.forward(input.long(), targets=None, is_eval=True)
            feature_magnitudes0 = output.feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)
            feature_magnitudes1 = output.feature_magnitudes[layer1]
            batchsize = feature_magnitudes0.shape[0]
            for b in range(batchsize):
                up = feature_magnitudes0[b:b+1].contiguous() #(1, seqlen, source_size)
                down = feature_magnitudes1[b:b+1] #(1, seqlen, targe_tsize)
                nz = (up > epsilon).nonzero() #(numnonzero, 3)
                for nonzeroidx in nz:
                    mask.fill_(1)
                    mask[nonzeroidx] = 0
                    result = forward(up*mask)
                    scores[nonzeroidx[1:]] += (result - down).abs().sum(dim=(0,1))
                    occurences[nonzeroidx[1:]] += 1
        return scores, occurences """


        
if __name__ == "__main__":
   #This code loads a model and data, and computes all the attributions
   #If you want to do you own run, just modify the strings here, and the arguments in the call to all_ig_attributions below
    c_name = 'staircasex8.shakespeare_64x4' #config options for the sae you want
    name = ''
    data_dir = 'data/shakespeare' #location of data, remember to prepare it!
    output_filename = 'Andy/data/direct_estimation.safetensors'
    batch_size = 32
    config = sae_options[c_name]

    model = SparsifiedGPT(config)
    model_path = os.path.join("checkpoints", name)
    model = model.load(model_path, device=config.device)
    model.to(config.device) #for some reason, when I do this the model starts on the cpu, and I have to move it


    #copied from Peter's training code
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = t.device(f"cuda:{ddp_local_rank}")

        assert t.cuda.is_available()
        t.cuda.set_device(device)
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = config.device

    dataloader = TrainingDataLoader(
        dir_path=data_dir,
        B= batch_size,
        T=model.config.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
    )
    x,_ = dataloader.next_batch(device)

    
    layers = model.gpt.config.n_layer

    out = {}
    for layer in range(layers):

        estimation, occ = direct_estimate(model, layer, layer+1, dataloader, nbatches=10)
        out[f'{layer}-{layer+1} scores'] = estimation
        out[f'{layer}-{layer+1} occurences'] = occ
        print(f'Finished layer {layer} to {layer+1}')

    save_file(out, output_filename)
    print(f'Saved to {output_filename}')
    
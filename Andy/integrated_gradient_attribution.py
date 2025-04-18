import sys
sys.path.append('/root/gpt-circuits')

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
from config.sae.models import sae_options, SAEVariant



from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from models.sae import SparseAutoencoder
from typing import Callable

from data.dataloaders import TrainingDataLoader
TensorFunction = Callable[[Tensor], Tensor]


def all_ig_attributions(model: SparsifiedGPT, ds: TrainingDataLoader, nbatches: int = 32, steps = 10, verbose = False, just_last = False):
    """
    Returns a dict of all consecutive integrated gradient attributions for a model. Most of the time, this is what you will call.
    :param model: SparsifiedGPT model
    :param ds: Dataloader
    :param nbatches: How many batches of data to aggregate into attributions
    :param steps: number of steps to approximate integral with
    :param verbose: Prints updates after finishing each layer connection
    :param just_last: whether to aggregate over all sequence positions or just take last
    :return: a dict where key 'i-i+1' (e.g. '0-1' or '1-2' is the 2d tensor of attributions between layers)
    TODO: give option to keep all sequence position
    """

    layers = model.gpt.config.n_layer
    attributions = {}
    for i in range(layers):
        attributions[f'{i}-{i+1}'] = ig_attributions(model, i, i+1, ds, nbatches, steps, just_last)
        ds.reset()
        if verbose:
            print(f"Finished Connections from Layer {i} to {i+1}")
    return attributions

def ig_attributions(model: SparsifiedGPT, layer0: int, layer1: int, ds: TrainingDataLoader, nbatches: int=32, steps = 10, just_last = False):
    """
    Computes integrated gradient attribution for a model between two layers
    :param model: SparsifiedGPT model
    :param layer0: index of layer of source sae
    :param layer1: index of layer of target sae
    :param ds: Dataloader
    :param nbatches: How many batches of data to aggregate into attributions
    :param steps: How many steps to use to approximate the integral
    :param just_last: If true, only does computation from the last position in the sequence
    
    :return: a tensor of shape (source_size, target_size) where the dimensions are the sizes of the hidden layers of the source and target sae respectively.
        If keep_pos is true, tensor is of shape (seq_len source_size seq_len target_size)
    """
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
    for param in forward.parameters():
        param.requires_grad = False

    source_size, _ = sae0.W_dec.shape
    target_size, _ = sae1.W_dec.shape
    
    attributions = t.zeros((source_size, target_size), device = model.gpt.config.device)
    is_jump = (sae0.config.sae_variant == SAEVariant.JUMP_RELU) or (sae0.config.sae_variant == SAEVariant.JUMP_RELU_STAIRCASE)
    

    for _ in range(nbatches):

        input, _ = ds.next_batch(model.gpt.config.device) #get batch of inputs 
        output = model.forward(input.long(), targets=None, is_eval=True)
        feature_magnitudes0 = output.feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)
        #if just_last:
            #feature_magnitudes0 = t.unsqueeze(feature_magnitudes0[:, -1, :], 1)
        if is_jump:
            threshold = t.exp(sae0.jumprelu.log_threshold) #(source_size)
            base = t.where(feature_magnitudes0 > 0, threshold, 0) #(batchsize, seqlen, source_size)
        else:
             base = None

        
        #loop over fms in target sae, and find attributions from all fms in source layer.
        #aggregate these by root mean square, following https://www.lesswrong.com/posts/Rv6ba3CMhZGZzNH7x/interpretability-integrated-gradients-is-a-decent
        for fm_i in range(target_size):
            y_i = t.zeros(target_size, device = model.gpt.config.device)
            y_i[fm_i] = 1
        
            gradient = integrate_gradient(x = feature_magnitudes0, x_i = None, fun = forward, direction = y_i, base = base, steps = steps, just_last = just_last) #(batch seq source_size)
            #sum over batch and position
            #Q: Does this make sense? It might be incorrect to sum over position
            if just_last:
                attributions[:,fm_i] = attributions[:,fm_i] + (gradient ** 2).sum(dim=0)
            else: 
                attributions[:,fm_i] = attributions[:,fm_i] + (gradient **2).sum(dim=[0,1])
                
    attributions = t.sqrt(attributions)
    return attributions

def integrate_gradient(x: Tensor, x_i: Tensor | None, fun: TensorFunction, direction, base = None, steps:int=10, just_last = False):
    """
    Approximates int_C d/dx_i y(z) dz where C is a linear path from base to x
    :param x: End of path. In practice it is a tensor of feature magnitudes generated by the data, 
        or a one hot encoding of a feature magnitude.
        Q: Should I only use certain x to compute attributions for a given x_i?.
        Shape (batchsize, seq_len, encoding)
    :param x_i: Direction of partial derivative. It is often a one hot encoding of a given feature magnitude. 
        If none, function computes and returns the attributions for all feature magnitudes 
        Shape (encoding)
    :param fun: Scalar valued function with signature (batchsize, seq_len, encoding) -> []. 
        In practice, it is the value of the jth feature magnitude after passing the feature magnitudes in the input layer through the model to the target layer
    :param base: Start of path. Default to 0
    :param steps: Number of steps to use to approximate the integral
    :return: x has shape (batch_size, seq_len, source_len). If x_i is not None, source_len is collapsed. If just_last, seq_len is collapsed

    Q: I do a lot to make this process memory light, because otherwise it crashes my pod. 
        Some of it is probably redundant/bad form

    """
    
    #compute a linear path from base to x
    if base is None:
        base = t.zeros_like(x)
    path = t.linspace(0, 1, steps)
    steplength = t.linalg.norm(x - base, dim = -1, keepdim = True)/steps


    batch_size, seq_len , _ = x.shape
    target_len = direction.shape[0]
    direction = direction.view(1, 1, target_len).expand(batch_size, seq_len, target_len)
    if just_last:
        direction[:,:-1,:] = 0

    integral = 0
    for alpha in path:
        point = (alpha*x + (1-alpha)*base).detach() #Find point on path
        point.requires_grad_()
        y = fun(point)  #compute gradient of y wrt x, scale by length of a step
        
        y.backward(retain_graph=False, gradient = direction) #we only need to do a backward pass once, this saves memory

        g = point.grad
        with t.no_grad(): #once again, no_grad is required to keep memory usage down
            if x_i == None:
                if just_last:
                    integral += g.detach()[:, -1, :] * steplength #(batchsize, source_len)
                else:
                    integral += g.detach() * steplength #(batchseize, seqlen, source_len)
                
            else:
                if just_last:
                    integral += (g.detach()[: :-1, :]* x_i).sum(dim=-1) * steplength #(batchsize)
                else:
                    integral += (g.detach() * x_i).sum(dim=-1) * steplength #(batchsize, seqlen)
    return integral

if __name__ == "__main__":
   #This code loads a model and data, and computes all the attributions
   #If you want to do you own run, just modify the strings here, and the arguments in the call to all_ig_attributions below
    c_name = "standardx16.tiny_32x4" #config options for the sae you want
    name = ''
    data_dir = 'data/tiny_stories_10m' #location of data, remember to prepare it!
    output_filename = 'Andy/data/standard_tiny.safetensors'
    batch_size = 32
    config = sae_options[c_name]

    model = SparsifiedGPT(config)
    model_path = os.path.join("checkpoints/standard.tiny_32x4", name)
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

    


    attributions1 = all_ig_attributions(model, dataloader, nbatches=32, steps=4, verbose = True)

    output_filename1 = 'Andy/data/standard_tiny_attributions.safetensors'
    save_file(attributions1, output_filename1)


    
    

   
    
 




   







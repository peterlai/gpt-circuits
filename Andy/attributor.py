import sys
sys.path.append('/workspace/gpt-circuits')

import torch as t
import torch.nn as nn
from torch import Tensor
from torch.autograd.functional import jacobian
import os

from safetensors.torch import save_file

import einops

from enum import Enum

from utils import MaxSizeList, get_SAE_activations

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from config.gpt.training import options
from config.sae.models import sae_options, SAEVariant



from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from models.sae import SparseAutoencoder
from typing import Callable

from data.dataloaders import TrainingDataLoader
TensorFunction = Callable[[Tensor], Tensor]

class PathType(Enum):
    BLOCK = "block"
    MLP = "MLP"

class Attributor():
    def __init__(self, model: nn.Module,  dataloader: TrainingDataLoader, nbatches: int = 32, verbose = False):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model. Most of the time, this is what you will call.
        :param model: SparsifiedGPT model
        :param ds: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param verbose: Prints updates after finishing each layer connection
        """
        self.model = model
        self.dataloader = dataloader
        self.nbatches = nbatches
        self.verbose = verbose
        self.attributions = {}

        if model.config.sae_variant.startswith('jsae') or model.config.sae_variant.startswith('mlp'):
            self.paths = PathType.MLP
        else:  
            self.paths = PathType.BLOCK
        
        

    def layer_by_layer(self)->dict:
        layers = self.model.gpt.config.n_layer
        if self.paths == PathType.BLOCK:
            for i in range(layers):
                self.attributions[f'{i}-{i+1}'] = self.single_layer(i, i+1)
                self.dataloader.reset()
                if self.verbose:
                    print(f"Finished Connections from Layer {i} to {i+1}")
            return self.attributions
        elif self.paths == PathType.MLP:
            for i in range(0, 2*layers, 2):
                self.attributions[f'{i}-{i+1}'] = self.single_layer(i, i+1)
                self.dataloader.reset()
                if self.verbose:
                    print(f"Finished Connections from Layer {i} to {i+1}")
            return self.attributions
    def single_layer(self, layer0, layer1):
        pass
    
    def make_computation_path(self, layer0, layer1):
        if self.paths == PathType.BLOCK:
            assert layer0 < layer1
            assert layer0 >= 0
            assert layer1 <= self.model.gpt.config.n_layer
            sae0 = self.model.saes[f'{layer0}']
            sae1 = self.model.saes[f'{layer1}']
            

            #define function that goes from feature magnitudes in layer0 to feature magnitudes in layer1
            #Q: Is this good form? I need it to make my Sequential object below
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)

            #construct function from Sae0 to Sae1
            forward_list = [Sae0Decode()] + [self.model.gpt.transformer.h[i] for i in range(layer0, layer1)] + [Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            return forward
        elif self.paths == PathType.MLP:
            assert layer0 + 1 ==  layer1
            assert layer0%2 == 0
            assert layer1 <= 2*self.model.gpt.config.n_layer
            sae0 = self.model.saes[f'{layer0}']
            sae1 = self.model.saes[f'{layer1}']
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)
            block = layer0//2
            forward_list = [Sae0Decode(), self.model.gpt.transformer.h[block].mlp, Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            return forward

            

            #define function that goes from feature magnitudes in layer0 to feature magnitudes in layer1
            #Q: Is this good form? I need it to make my Sequential object below
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)

            #construct function from Sae0 to Sae1
            forward_list = [Sae0Decode()] + [self.model.gpt.transformer.h[i] for i in range(layer0, layer1)] + [Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            return forward

class IntegratedGradientAttributor(Attributor):
    def __init__(self, model: nn.Module,  dataloader: TrainingDataLoader, nbatches: int = 32, steps = 10, verbose = False, abs = False, just_last = False):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model. Most of the time, this is what you will call.
        :param model: SparsifiedGPT model
        :param ds: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param steps: number of steps to approximate integral with
        :param verbose: Prints updates after finishing each layer connection
        :param just_last: whether to aggregate over all sequence positions or just take last
        """
        super().__init__(model, dataloader, nbatches, verbose)

        self.steps = steps
        self.abs = abs
        self.just_last = just_last

        self.is_jump = (self.model.saes['0'].config.sae_variant == SAEVariant.JUMP_RELU) or (self.model.saes['0'].config.sae_variant == SAEVariant.JUMP_RELU_STAIRCASE)

    def single_layer(self, layer0, layer1):

        forward = self.make_computation_path(layer0, layer1)
        for param in forward.parameters():
            param.requires_grad = False

        sae0 = self.model.saes[f'{layer0}']
        sae1 = self.model.saes[f'{layer1}']

        source_size, _ = sae0.W_dec.shape
        target_size, _ = sae1.W_dec.shape
        
        attributions = t.zeros((source_size, target_size), device = self.model.gpt.config.device)
        
        

        for _ in range(self.nbatches):

            input, _ = self.dataloader.next_batch(self.model.gpt.config.device) #get batch of inputs 
            #output = self.model.forward(input.long(), targets=None, is_eval=True)
            feature_magnitudes = get_SAE_activations(self.model, input.long(), [layer0, layer1])
            feature_magnitudes0 = feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)

            if self.is_jump:
                threshold = t.exp(sae0.jumprelu.log_threshold) #(source_size)
                base = t.where(feature_magnitudes0 > 0, threshold, 0) #(batchsize, seqlen, source_size)
            else:
                base = None

            
            #loop over fms in target sae, and find attributions from all fms in source layer.
            #aggregate these by root mean square, following https://www.lesswrong.com/posts/Rv6ba3CMhZGZzNH7x/interpretability-integrated-gradients-is-a-decent
            for fm_i in range(target_size):
                y_i = t.zeros(target_size, device = self.model.gpt.config.device)
                y_i[fm_i] = 1
            
                gradient = self.integrate_gradient(x = feature_magnitudes0, x_i = None, fun = forward, direction = y_i, base = base) #(batch seq source_size)
                #sum over batch and position

                if self.abs:
                    if self.just_last:
                        attributions[:,fm_i] = attributions[:,fm_i] + (gradient.abs()).sum(dim=0)
                    else: 
                        attributions[:,fm_i] = attributions[:,fm_i] + (gradient.abs()).sum(dim=[0,1])
                else:
                    if self.just_last:
                        attributions[:,fm_i] = attributions[:,fm_i] + (gradient**2).sum(dim=0)
                    else: 
                        attributions[:,fm_i] = attributions[:,fm_i] + (gradient**2).sum(dim=[0,1])
                    
                    attributions = t.sqrt(attributions)
        return attributions

    def integrate_gradient(self, x: Tensor, x_i: Tensor | None, fun: TensorFunction, direction, base = None):
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
        :return: x has shape (batch_size, seq_len, source_len). If x_i is not None, source_len is collapsed. If just_last, seq_len is collapsed

        Q: I do a lot to make this process memory light, because otherwise it crashes my pod. 
            Some of it is probably redundant/bad form

        """
        
        #compute a linear path from base to x
        if base is None:
            base = t.zeros_like(x)
        path = t.linspace(0, 1, self.steps)
        steplength = t.linalg.norm(x - base, dim = -1, keepdim = True)/self.steps


        batch_size, seq_len , _ = x.shape
        target_len = direction.shape[0]
        direction = direction.view(1, 1, target_len).expand(batch_size, seq_len, target_len)
        if self.just_last:
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
                    if self.just_last:
                        integral += g.detach()[:, -1, :] * steplength #(batchsize, source_len)
                    else:
                        integral += g.detach() * steplength #(batchseize, seqlen, source_len)
                    
                else:
                    if self.just_last:
                        integral += (g.detach()[: :-1, :]* x_i).sum(dim=-1) * steplength #(batchsize)
                    else:
                        integral += (g.detach() * x_i).sum(dim=-1) * steplength #(batchsize, seqlen)
        return integral

class ManualAblationAttributor(Attributor):
    def __init__(self, model: nn.Module,  dataloader: TrainingDataLoader, nbatches: int = 32, verbose = True, epsilon=0, max_size: int=10000):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model. Most of the time, this is what you will call.
        :param model: SparsifiedGPT model
        :param ds: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param verbose: Prints updates after finishing each layer connection
        """
        super().__init__(model, dataloader, nbatches, verbose)
        self.epsilon = epsilon
        self.max_size = max_size

    def single_layer(self, layer0, layer1):
        with t.no_grad():
            
            forward = self.make_computation_path(layer0, layer1)

            source_size, _ = sae0.W_dec.shape
            target_size, _ = sae1.W_dec.shape

            scores = t.zeros((source_size, target_size), device = self.model.gpt.config.device)
            occurences = t.zeros((self.model.gpt.config.block_size, source_size), device = self.model.gpt.config.device)


            for _ in range(nbatches):
                input, _ = self.dataloader.next_batch(self.model.gpt.config.device) #get batch of inputs 
                #output = self.model.forward(input.long(), targets=None, is_eval=True)
                feature_magnitudes = get_SAE_activations(self.model, input.long(), [layer0, layer1])
                feature_magnitudes0 = output.feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)
                feature_magnitudes1 = output.feature_magnitudes[layer1] #feature magnitudes at target layer (batchsize, seqlen, target_size)
                batchsize = feature_magnitudes0.shape[0]
                #epsilon = feature_magnitudes0.mean().item()

                batch_patches = MaxSizeList(max_size)  # List to store patches
                batch_indices = MaxSizeList(max_size)  # List to store indices

                for b in range(batchsize):
                    up = feature_magnitudes0[b:b+1].contiguous()  # (1, seqlen, source_size)
                    down = feature_magnitudes1[b:b+1]  # (1, seqlen, target_size)
                    nz = (up > epsilon).nonzero(as_tuple=True)  # Get indices as tuples
                    for idx in zip(*nz):
                        patch = up.clone()  # Clone once per batch element
                        patch[0, idx[1], idx[2]] = 0  # Modify in-place
                        batch_patches.append((patch, up[0, idx[1], idx[2]]))  # Append the patch and the original value
                        batch_indices.append(((b, idx[1], idx[2]), up[0, idx[1], idx[2]])) # Append the index and the original value
                batch_patches = batch_patches._get_valueless_list()  # Get the list of patches
                batch_indices = batch_indices._get_valueless_list()  # Get the list of indices

                # Run a single forward pass for all patches
                if batch_patches:
                    #print(len(batch_patches))
                    batch_patches = t.cat(batch_patches, dim=0) # Concatenate patches into a single tensor (num_patches, seq_len, source_size)
                    #print(batch_patches.shape)  
                    results = forward(batch_patches)  # Forward pass for all patches (num_patches, seq_len, target_size)

                    # Update scores and occurrences
                    for i, (b, seq_idx, src_idx) in enumerate(batch_indices):
                        scores[src_idx] += (results[i] - feature_magnitudes1[b]).abs().sum(dim=(0))
                        occurences[seq_idx, src_idx] += 1

                    # Clear the batch lists
                    # This is important to avoid memory leaks
                    del batch_patches
                    del batch_indices
                    del results
                    t.cuda.empty_cache()

            #I currently track occurences. I could normalize here, though it might be computation inefficient
            return scores
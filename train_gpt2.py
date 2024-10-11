import csv
import os
import math
import time
import inspect
from dataclasses import dataclass
from typing import Iterable, Optional
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------


@dataclass
class ModelOutput:
    logits: torch.Tensor
    gpt_loss: torch.Tensor = torch.Tensor()
    sae_reconstruction_losses: torch.Tensor = torch.Tensor()
    sae_sparsity_losses: torch.Tensor = torch.Tensor()
    sae_aux_losses: torch.Tensor = torch.Tensor()
    sae_l0_losses: torch.Tensor = torch.Tensor()


@dataclass
class SAEForwardOutput:
    """
    Output from the forward pass of an SAE model.
    """

    reconstructed_activations: torch.Tensor
    feature_magnitudes: torch.Tensor
    reconstruct_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    aux_loss: torch.Tensor
    l0: torch.Tensor


class JumpReLUSAE(nn.Module):
    """
    JumpRelU Autoencoder module.
    https://arxiv.org/pdf/2407.14435

    Derived from:
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(self, config, layer_idx):
        """
        n_embd: GPT embedding size.
        n_sae: SAE dictionary size.
        """
        super().__init__()
        self.sparsity_coefficient = config.sparsity_coefficient
        self.bandwidth = config.bandwidth
        self.b_dec = nn.Parameter(torch.zeros(config.n_embd))
        self.b_enc = nn.Parameter(torch.zeros(config.n_sae))
        # TODO: Do we need to unit normalize the columns of W_enc?
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(config.n_embd, config.n_sae)))
        self.W_dec = nn.Parameter(self.W_enc.mT.detach().clone())
        self.log_threshold = nn.Parameter(torch.full((config.n_sae,), math.log(self.bandwidth)))

    def encode(self, x: torch.Tensor, gain: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        threshold = torch.exp(self.log_threshold)
        feature_magnitudes: torch.Tensor = JumpReLUFunction.apply(pre_activations, threshold, self.bandwidth)  # type: ignore

        if gain is not None:
            feature_magnitudes = feature_magnitudes * gain

        return feature_magnitudes

    def decode(self, x: torch.Tensor):
        """
        x: SAE activations (batch_size, n_sae)
        """
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, gain: Optional[torch.Tensor] = None) -> SAEForwardOutput:
        """
        Returns (i) a reconstruction of GPT model activations, (ii) the SAE activations, and (iii) SAE loss components.

        x: GPT model activations (batch_size, n_embd)
        """
        feature_magnitudes = self.encode(x, gain)
        x_reconstructed = self.decode(feature_magnitudes)

        # L2 reconstruction loss
        reconstruction_loss = F.mse_loss(x_reconstructed, x)

        # L0 sparsity loss
        threshold = torch.exp(self.log_threshold)
        l0 = (
            StepFunction.apply(
                feature_magnitudes,
                threshold,
                self.bandwidth,
            )
            .sum(dim=-1)  # type: ignore
            .mean()
        )
        sparsity_loss = l0 * self.sparsity_coefficient
        aux_loss = torch.zeros_like(sparsity_loss)

        return SAEForwardOutput(x_reconstructed, feature_magnitudes, reconstruction_loss, sparsity_loss, aux_loss, l0)


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def rectangle(x) -> torch.Tensor:
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, output_grad):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold) * output_grad
        threshold_grad = (
            -(threshold / bandwidth) * JumpReLUFunction.rectangle((x - threshold) / bandwidth) * output_grad
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def rectangle(x) -> torch.Tensor:
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def forward(ctx, x, threshold, bandwidth) -> torch.Tensor:
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(grad_output)
        threshold_grad = -(1.0 / bandwidth) * StepFunction.rectangle((x - threshold) / bandwidth) * grad_output
        return x_grad, threshold_grad, None  # None for bandwidth


class BaseGatedSAE(nn.Module):
    """
    Gated Sparse Autoencoder module.
    https://arxiv.org/abs/2404.16014
    """

    def __init__(self, config, layer_idx):
        """
        n_embd: GPT embedding size.
        n_sae: SAE dictionary size.
        """
        super().__init__()
        self.l1_coefficient = config.l1_coefficients[layer_idx]
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(config.n_sae, config.n_embd)))
        self.b_gate = nn.Parameter(torch.zeros(config.n_sae))
        self.b_mag = nn.Parameter(torch.zeros(config.n_sae))
        self.b_dec = nn.Parameter(torch.zeros(config.n_embd))

    def get_W_gate(self):
        """
        To be implemented by derived classes.
        """
        raise NotImplementedError()

    def get_W_mag(self):
        """
        To be implemented by derived classes.
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor, gain: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pi_gate = x_centered @ self.get_W_gate() + self.b_gate

        f_gate = (pi_gate > 0).float()  # whether to gate the feature
        f_mag = F.relu(x_centered @ self.get_W_mag() + self.b_mag)  # feature magnitudes

        x_encoded = f_gate * f_mag

        if gain is not None:
            x_encoded = x_encoded * gain

        return x_encoded, pi_gate

    def decode(self, x: torch.Tensor):
        """
        x: SAE activations (batch_size, n_sae)
        """
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, gain: Optional[torch.Tensor] = None) -> SAEForwardOutput:
        """
        Returns (i) a reconstruction of GPT model activations, (ii) the SAE activations, and (iii) SAE loss components.

        x: GPT model activations (batch_size, n_embd)
        """
        feature_magnitudes, pi_gate = self.encode(x, gain)
        x_reconstructed = self.decode(feature_magnitudes)

        # L2 reconstruction loss
        reconstruction_loss = F.mse_loss(x_reconstructed, x)

        # Use Gated (RI-L1) sparsity variant: https://arxiv.org/pdf/2407.14435
        scaled_pi_gate = F.relu(pi_gate) * self.W_dec.data.norm(dim=1)
        sparsity_loss = F.l1_loss(scaled_pi_gate, torch.zeros_like(pi_gate)) * self.l1_coefficient

        # compute the auxiliary loss
        W_dec_clone = self.W_dec.clone().detach()
        b_dec_clone = self.b_dec.clone().detach()
        x_hat_frozen = nn.ReLU()(pi_gate) @ W_dec_clone + b_dec_clone
        aux_loss = F.mse_loss(x_hat_frozen, x)

        # L0 sparsity loss
        l0 = (feature_magnitudes != 0).sum(dim=-1).float().mean()

        return SAEForwardOutput(x_reconstructed, feature_magnitudes, reconstruction_loss, sparsity_loss, aux_loss, l0)


class GatedSAE(BaseGatedSAE):
    """
    Standed Gated Sparse Autoencoder module (RI-L1).
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.W_gate = nn.Parameter(self.W_dec.mT.detach().clone())
        self.r_mag = nn.Parameter(torch.zeros(config.n_sae))

    def get_W_gate(self):
        return self.W_gate

    def get_W_mag(self):
        return self.get_W_gate() * torch.exp(self.r_mag)


class GatedSAE_V2(BaseGatedSAE):
    """
    Experimental Gated Sparse Autoencoder module that ties the encoder and decoder weights to avoid feature absorption.
    Reference: https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN
    """

    def get_W_gate(self):
        """
        Tying encoder weights to decoder weights.
        """
        return self.W_dec.t()

    def get_W_mag(self):
        """
        The r_mag scaler doesn't seem useful after weights are tied.
        """
        return self.get_W_gate()


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.sae = GatedSAE_V2(config, layer_idx)

    def forward(self, x) -> tuple[torch.Tensor, SAEForwardOutput]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        sae_output = self.sae(x)
        return x, sae_output


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_sae: int = n_embd * 16 # number of SAE features
    l1_coefficients: tuple = (1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.) # SAE L1 regularization coefficients


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, layer_idx=i+1) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.sae = GatedSAE_V2(config, layer_idx=0)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        sae_reconstruction_losses_layers = []
        sae_sparsity_losses_layers = []
        sae_aux_losses_layers = []
        sae_l0_losses_layers = []

        embedding_sae_output = self.sae(x)
        sae_reconstruction_losses_layers.append(embedding_sae_output.reconstruct_loss)
        sae_sparsity_losses_layers.append(embedding_sae_output.sparsity_loss)
        sae_aux_losses_layers.append(embedding_sae_output.aux_loss)
        sae_l0_losses_layers.append(embedding_sae_output.l0)

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x, block_sae_output = block(x)
            sae_reconstruction_losses_layers.append(block_sae_output.reconstruct_loss)
            sae_sparsity_losses_layers.append(block_sae_output.sparsity_loss)
            sae_aux_losses_layers.append(block_sae_output.aux_loss)
            sae_l0_losses_layers.append(block_sae_output.l0)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            gpt_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            return ModelOutput(
                logits=logits,
                gpt_loss=gpt_loss,
                sae_reconstruction_losses=torch.stack(sae_reconstruction_losses_layers),
                sae_sparsity_losses=torch.stack(sae_sparsity_losses_layers),
                sae_aux_losses=torch.stack(sae_aux_losses_layers),
                sae_l0_losses=torch.stack(sae_l0_losses_layers)
            )

        return ModelOutput(logits=logits)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
csv_file = open(os.path.join(log_dir, f"{int(time.time())}.csv"), "w")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "type", "step", "gpt", "sae_reconstruction", "sae_sparsity", "sae_aux", "sae_l0", "loss", "hella", "lr", "norm", "dt", "tok/sec"
])
csv_writer.writeheader()

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_gpt_loss_accum = torch.tensor(0., device=device)
            val_sae_reconstruction_accum = torch.tensor(0., device=device)
            val_sae_sparsity_accum = torch.tensor(0., device=device)
            val_sae_aux_accum = torch.tensor(0., device=device)
            val_sae_l0_accum = torch.tensor(0., device=device)
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_output: ModelOutput = model(x, y)
                val_gpt_loss_accum += (model_output.gpt_loss / val_loss_steps).detach()
                val_sae_reconstruction_accum += (model_output.sae_reconstruction_losses.mean() / val_loss_steps).detach()
                val_sae_sparsity_accum += (model_output.sae_sparsity_losses.mean() / val_loss_steps).detach()
                val_sae_aux_accum += (model_output.sae_aux_losses.mean() / val_loss_steps).detach()
                val_sae_l0_accum += (model_output.sae_l0_losses.mean() / val_loss_steps).detach()

        if ddp:
            dist.all_reduce(val_gpt_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_sae_reconstruction_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_sae_sparsity_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_sae_aux_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_sae_l0_accum, op=dist.ReduceOp.AVG)

        val_loss_accum = val_gpt_loss_accum + val_sae_reconstruction_accum + val_sae_sparsity_accum + val_sae_aux_accum

        if master_process:
            log_data = {
                "type": "eval",
                "step": step,
                "gpt": round(val_gpt_loss_accum.item(), 6),
                "sae_reconstruction": round(val_sae_reconstruction_accum.item(), 6),
                "sae_sparsity": round(val_sae_sparsity_accum.item(), 6),
                "sae_aux": round(val_sae_aux_accum.item(), 6),
                "sae_l0": round(val_sae_l0_accum.item(), 6),
                "loss": round(val_loss_accum.item(), 6),
            }
            csv_writer.writerow(log_data)
            csv_file.flush()
            print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint_metadata = log_data.copy()
                del checkpoint_metadata["type"]
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    **checkpoint_metadata,
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_output: ModelOutput = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, model_output.logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            log_data = {
                "type": "eval",
                "step": step,
                "hella": acc_norm,
            }
            csv_writer.writerow(log_data)
            csv_file.flush()

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_output: ModelOutput = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = model_output.logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    gpt_loss_accum = torch.tensor(0., device=device)
    sae_reconstruction_accum = torch.tensor(0., device=device)
    sae_sparsity_accum = torch.tensor(0., device=device)
    sae_aux_accum = torch.tensor(0., device=device)
    sae_l0_accum = torch.tensor(0., device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            model_output: ModelOutput = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        gpt_loss = model_output.gpt_loss / grad_accum_steps
        # Average SAE loss components over the layers
        sae_reconstruction_loss = model_output.sae_reconstruction_losses.mean() / grad_accum_steps
        sae_sparsity_loss = model_output.sae_sparsity_losses.mean() / grad_accum_steps
        sae_aux_loss = model_output.sae_aux_losses.mean() / grad_accum_steps
        sae_l0_loss = model_output.sae_l0_losses.mean() / grad_accum_steps

        gpt_loss_accum += gpt_loss.detach()
        sae_reconstruction_accum += sae_reconstruction_loss.detach()
        sae_sparsity_accum += sae_sparsity_loss.detach()
        sae_aux_accum += sae_aux_loss.detach()
        sae_l0_accum += sae_l0_loss.detach()

        loss = gpt_loss + sae_reconstruction_loss + sae_sparsity_loss + sae_aux_loss
        loss.backward()

    if ddp:
        dist.all_reduce(gpt_loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(sae_reconstruction_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(sae_sparsity_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(sae_aux_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(sae_l0_accum, op=dist.ReduceOp.AVG)

    loss_accum = gpt_loss_accum + sae_reconstruction_accum + sae_sparsity_accum + sae_aux_accum

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        log_data = {
            "type": "train",
            "step": step,
            "gpt": round(gpt_loss_accum.item(), 6),
            "sae_reconstruction": round(sae_reconstruction_accum.item(), 6),
            "sae_sparsity": round(sae_sparsity_accum.item(), 6),
            "sae_aux": round(sae_aux_accum.item(), 6),
            "sae_l0": round(sae_l0_accum.item(), 6),
            "loss": round(loss_accum.item(), 6),
            "lr": f"{lr:.4e}",
            "norm": round(norm.item(), 4),
            "dt": round(dt, 4),
            "tok/sec": round(tokens_per_sec, 2),
        }
        csv_writer.writerow(log_data)
        csv_file.flush()

        print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

if ddp:
    destroy_process_group()

csv_file.close()

"""
GPT-2 model. Adopted from: https://github.com/karpathy/build-nanogpt
"""

import dataclasses
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.nn import functional as F

from config.gpt.models import GPTConfig


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config, layer_idx=i + 1) for i in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        for name, module in self.named_modules():
            self._init_weights(name, module)

    def _init_weights(self, name, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if name.endswith(".c_proj"):
                # "Weights of residual layers are scaled at initialization by a factor of 1/√N
                # where N is the number of residual layers."
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache_resid=False):
        # idx is of shape (B, T)
        cache = []

        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        if cache_resid:
            cache.append(x.detach().clone()) # (batch, seq, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            if cache_resid:
                cache.append(x.detach().clone())

        if cache_resid:
            mycache = torch.stack(cache, dim=1) # (batch, seq, n_embd) -> (batch, n_layer+1, seq,  n_embd)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            if cache_resid:
                return (logits, loss, mycache)
            else:
                return (logits, loss)
            
        if cache_resid:
            return (logits, None, mycache)
        else:
            return (logits, None)

    def forward_with_patched_activations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass of the model with patched activations.

        :param x: Patched activations. Shape: (B, T, n_embd)
        :param layer_idx: Layer index. 0 patches activations just before the first transformer block.
        """
        # forward through transformer blocks starting with the specified layer
        for block in self.transformer.h[layer_idx:]:
            x = block(x)

        # forward through the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def load(cls, dir, device: torch.device):
        meta_path = os.path.join(dir, "model.json")
        weights_path = os.path.join(dir, "model.safetensors")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model = GPT(GPTConfig(**meta))

        load_model(model, weights_path, device=device.type)
        return model

    def save(self, dir):
        meta_path = os.path.join(dir, "model.json")
        weights_path = os.path.join(dir, "model.safetensors")

        meta = dataclasses.asdict(self.config, dict_factory=GPTConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        save_model(self, weights_path)

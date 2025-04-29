"""
GPT-2 model. Adopted from: https://github.com/karpathy/build-nanogpt
"""

import dataclasses
import json
import os
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.nn import functional as F
from torch import Tensor
from config.gpt.models import GPTConfig
from jaxtyping import Float, Int

import warnings

from collections import namedtuple

GPTOutput = namedtuple("GPTOutput", ["logits", "loss"])

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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
    @property
    def W_in(self) -> torch.Tensor:
        return self.c_fc.weight
    
    @property
    def W_out(self) -> torch.Tensor:
        return self.c_proj.weight


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, resid_pre : torch.Tensor) -> torch.Tensor:
        resid_mid = resid_pre + self.attn(self.ln_1(resid_pre))
        resid_post = resid_mid + self.mlp(self.ln_2(resid_mid)) #can capture resid_mid via input to ln_2
        return resid_post


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


    def embed(self, idx : Int[Tensor, "B T"]) -> Float[Tensor, "B T n_embd"]:
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        resid = tok_emb + pos_emb
        return resid

    
    def forward(self,
                input : Union[Int[Tensor, "B T"], Float[Tensor, "B T n_embd"]],
                targets : Optional[Int[Tensor, "B T"]]=None,
                start_at_layer : Optional[int]=None,
                stop_at_layer : Optional[int]=None
        ) -> Union[Float[Tensor, "B T n_embd"], GPTOutput]: # NOTE: Assumes GPTOutput namedtuple is defined elsewhere
        # idx is of shape (B, T)
        """
        Forward pass of the GPT model.
        start_at_layer and stop_at_layer are optional parameters that allow the forward pass to be
        stopped at a specific layer. If not specified, the full model is run.
        See https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformer.py
        
        Args:
            input: Input tensor. Can be token indices (Int[Tensor, "B T"]) or embeddings
                   (Float[Tensor, "B T n_embd"]) if start_at_layer is specified.
            targets: Optional target tensor for loss calculation. Shape: (Int[Tensor, "B T"]).
            start_at_layer: Optional layer index to start the forward pass from. Inclusive.
                            Requires input to be embeddings. Defaults to None (start from embedding).
            stop_at_layer: Optional layer index to stop the forward pass at. Exclusive.
                           If specified, returns the residual stream at that point. Defaults to None (run full model).

        Returns:
            Union[Float[Tensor, "B T n_embd"], GPTOutput]:
                - If stop_at_layer is not None, returns the residual stream tensor of shape (B, T, n_embd).
                - Otherwise, returns a GPTOutput namedtuple containing:
                    - logits: Output logits tensor of shape (B, T, vocab_size).
                    - loss: Calculated cross-entropy loss (scalar tensor) if targets are provided, else None.

        start_at_layer Optional[int]: If not None, start the forward pass at the specified
                layer. Requires input to be the residual stream before the specified layer with
                shape [batch, pos, d_model]. Inclusive - ie, start_at_layer = 0 skips the embedding
                then runs the rest of the model. Supports negative indexing. start_at_layer = -1
                only runs the final block and the unembedding. Defaults to None (run the full
                model).
                
         stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer.
                Exclusive - ie, stop_at_layer = 0 will only run the embedding layer, stop_at_layer =
                1 will run the embedding layer and the first transformer block, etc. Supports
                negative indexing. Useful for analysis of intermediate layers, eg finding neuron
                activations in layer 3 of a 24 layer model. Defaults to None (run the full model).
                If not None, we return the last residual stream computed.
        """
        if start_at_layer is None:
            assert input.ndim == 2 and not input.is_floating_point(), f"input must be an integer tensor of shape (B, T) when start_at_layer is None"
            resid = self.embed(input)
            start_at_layer = 0
        else:
            assert input.ndim == 3 and input.is_floating_point(), f"input must be a float tensor of shape (B, T, n_embd) when start_at_layer is specified"
            resid = input

        if start_at_layer < 0:
            start_at_layer = self.config.n_layer + start_at_layer

        if stop_at_layer is not None and stop_at_layer < 0:
             stop_at_layer = self.config.n_layer + stop_at_layer

        for block in self.transformer.h[start_at_layer:stop_at_layer]:
            resid = block(resid)

        if stop_at_layer is not None:
            if targets is not None:
                warnings.warn("GPT.forward: Cannot measure loss if stop_at_layer is used. Returning last residual stream.")
            return resid

        # forward the final layernorm and the classifier
        resid = self.transformer.ln_f(resid)
        logits = self.lm_head(resid)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return GPTOutput(logits=logits, loss=loss)

    def forward_with_patched_activations(self, resid: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.forward(resid, start_at_layer=layer_idx).logits

    @classmethod
    def load(cls, dir, device: torch.device):
        meta_path = os.path.join(dir, "model.json")
        weights_path = os.path.join(dir, "model.safetensors")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        model = cls(GPTConfig(**meta))

        load_model(model, weights_path, device=device.type)
        return model

    def save(self, dir):
        meta_path = os.path.join(dir, "model.json")
        weights_path = os.path.join(dir, "model.safetensors")

        meta = dataclasses.asdict(self.config, dict_factory=GPTConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        save_model(self, weights_path)

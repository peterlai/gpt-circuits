# %%
%load_ext autoreload
%autoreload 2       
# %%
# from dataclasses import dataclass
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Change current working directory to parent
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())
# %%
import torch
from models.gpt import MLP
from models.sae.topk import TopKSAE
from models.sae import SAEConfig


from pathlib import Path
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.jsaesparsified import JSparsifiedGPT

from typing import Optional, Literal
from torch.nn import functional as F
import einops

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients()

gpt_dir = Path("checkpoints/shakespeare_64x4")
gpt = GPT.load(gpt_dir, device=device)

sae_config =SAEConfig(
        gpt_config=gpt.config,
        n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
        sae_variant=SAEVariant.TOPK,
        top_k = (10, 10, 10, 10, 10, 10, 10, 10)
    )

gpt_mlp = JSparsifiedGPT.load("checkpoints/jsae.shakespeare_64x4", 
                                 loss_coefficients,
                                 trainable_layers = None,
                                 device = device)
gpt_mlp = torch.compile(gpt_mlp)
gpt_mlp = gpt_mlp.to(device)
#tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)    

# %%
from torch import Tensor
from jaxtyping import Float, Int

def original_jacobian(
    sae_mlpin : TopKSAE,
    sae_mlpout : TopKSAE,
    mlp: MLP,
    topk_indices_mlpin: torch.Tensor,
    topk_indices_mlpout: torch.Tensor,
    mlp_act_grads: torch.Tensor,
) -> torch.Tensor:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)
    
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)

    dtype = wd1.dtype

    jacobian = einops.einsum(
        wd1[topk_indices_mlpin],
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_mlpout],
        # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
        # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    )
    return jacobian

@torch.compile(mode="max-autotune", fullgraph=True)
def jacobian_fast(
    sae_mlpin : TopKSAE,
    sae_mlpout : TopKSAE,
    mlp: MLP,
    topk_indices_mlpin: Int[Tensor, "batch seq k"],
    topk_indices_mlpout: Int[Tensor, "batch seq k"],
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"],
) -> torch.Tensor:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)
    
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)

    dtype = wd1.dtype

    jacobian = einops.einsum(
        wd1[topk_indices_mlpin], #(batch, seq, k_in, d_mlp)
        mlp_act_grads.to(dtype), #(batch, seq, d_mlp)
        w2e[:, topk_indices_mlpout], #(d_mlp, batch, seq, k_out)
        # "... seq k1 d_mlp, ... seq d_mlp,"
        # "d_mlp ... seq k2 -> ... seq k2 k1",
        "... k_in d_mlp, ... d_mlp, d_mlp ... k_out -> ... k_out k_in", # ... = batch, seq
    )
    k_in = sae_mlpin.k
    k_out = sae_mlpout.k
    assert k_in == k_out
    return jacobian


sae_mlpin = gpt_mlp.saes['0_mlpin']
sae_mlpout = gpt_mlp.saes['0_mlpout']
mlp = gpt_mlp.gpt.transformer.h[0].mlp



# %%

import time

activations = {}
def mlp_gelu_hook_fn(module, inputs, outputs):
    pre_actfn = inputs[0]
    post_actfn = outputs

    pre_actfn_copy = pre_actfn.detach().requires_grad_(True)

    with torch.enable_grad():
        recomputed_post_actfn = F.gelu(pre_actfn_copy)
        grad_of_actfn = torch.autograd.grad(
            outputs=recomputed_post_actfn,
            inputs=pre_actfn_copy,
            grad_outputs=torch.ones_like(recomputed_post_actfn),
            retain_graph=False,
            create_graph=False
        )[0]

    activations["mlp_gelu"] = grad_of_actfn
    return outputs

# Hook registration
hook_handle = mlp.gelu.register_forward_hook(mlp_gelu_hook_fn)

batch, seq, d_model, feat_size = 1, 1, 64, 512


random_features = torch.randn((batch, seq, feat_size), device=device)

top_k_values, topk_indices_pre = torch.topk(random_features, sae_mlpin.k, dim=-1)
mask = random_features >= top_k_values[..., -1].unsqueeze(-1)
feat_pre_mlp = random_features * mask.float()

resid_pre_mlp = sae_mlpin.decode(feat_pre_mlp)
resid_post_mlp = mlp(resid_pre_mlp)

feat_post_mlp, topk_indices_post = sae_mlpout.encode(resid_post_mlp, return_topk_indices=True)


jacobian = jacobian_fast(
    sae_mlpin,
    sae_mlpout,
    mlp,
    topk_indices_pre,
    topk_indices_post,
    activations["mlp_gelu"]
)
# %%
def untopk(sparse_feat_mags : Float[Tensor, "batch seq k"], 
             indices : Int[Tensor, "batch seq k"],
             feature_size : int):
    """
    undoes the topk operation, padding with zeros
    """
    assert sparse_feat_mags.shape == indices.shape
    batch, seq, k = sparse_feat_mags.shape
    feat_mags = torch.zeros((batch, seq, feature_size), 
                            dtype=sparse_feat_mags.dtype, 
                            device=sparse_feat_mags.device)
    feat_mags.scatter_(dim=-1, index=indices, src=sparse_feat_mags)
    return feat_mags


def sandwich(sparse_feat_mags : Float[Tensor, "batch seq k"], indices : Int[Tensor, "batch seq k"]):
    """
    takes sparse feature magnitudes and indices, and returns the sandwich product
    """
    batch, seq, k = sparse_feat_mags.shape
    sae_mlpin_featmag = untopk(sparse_feat_mags, indices, sae_mlpin.feature_size)
    
    mlp_pre_act = sae_mlpin.decode(sae_mlpin_featmag)
    mlp_post_act = mlp(mlp_pre_act)
    
    sae_mlpin_featmag = sae_mlpout.encode(mlp_post_act)
    sparse_feat_mags_out, topk_indices_out = torch.topk(sae_mlpin_featmag, 10, dim=-1)
    return sparse_feat_mags_out, topk_indices_out
  
# %%

batch, seq, k = 1, 1, 10
sparse_feat_mags_in = torch.randn((batch, seq, k), device=device)
topk_indices_in = torch.arange(k, device=device, dtype=torch.int64)
topk_indices_in = einops.repeat(topk_indices_in, "k -> b s k", b=batch, s=seq)
sparse_feat_mags_out, topk_indices_out = sandwich(sparse_feat_mags_in, topk_indices_in)

jacobian = jacobian_fast(
    sae_mlpin,
    sae_mlpout,
    mlp,
    topk_indices_in,
    topk_indices_out,
    activations["mlp_gelu"]
)

sparse_feat_mags_in_bump = sparse_feat_mags_in + 0.1
sparse_feat_mags_out_bump, topk_indices_out_bump = sandwich(sparse_feat_mags_in_bump, topk_indices_in)

estimate_sparse_feat_mags_out_bump = sparse_feat_mags_out + jacobian @ torch.ones_like(sparse_feat_mags_in) * 0.1





# %%



# %%

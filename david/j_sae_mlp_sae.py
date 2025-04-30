# %%
# %load_ext autoreload
# %autoreload 2
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
from models.gpt import GPTConfig

from pathlib import Path
from config.sae.models import SAEConfig, SAEVariant

from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.jsaesparsified import JSparsifiedGPT

from typing import Optional, Literal
from torch.nn import functional as F
import einops

from jaxtyping import Float, Int
from torch import Tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn as nn



class DummySAE(nn.Module):
    def __init__(self, n_embd: int, n_features: int):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd)))
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_enc
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec

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


def all_jacobian(
    sae_mlpin : TopKSAE,
    sae_mlpout : TopKSAE,
    mlp: MLP,
    mlp_act_grads: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Jacobian of the output with respect to the input for all possible input indices.
    """
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)

    dtype = wd1.dtype

    jacobian = einops.einsum(
        wd1,
        mlp_act_grads.to(dtype),
        w2e,
        "... feat1 d_mlp, ... d_mlp, d_mlp ... feat2 -> ... feat2 feat1",
    )
    return jacobian


def sandwich(in_feat_mags : Float[Tensor, "batch seq feat"],
             sae_mlpin : DummySAE,
             mlp : MLP,
             sae_mlpout : DummySAE):
    """
    takes sparse feature magnitudes and indices, and returns the sandwich product
    """
    
    mlp_pre_act = sae_mlpin.decode(in_feat_mags)
    mlp_post_act = mlp(mlp_pre_act)
    sae_mlpin_featmag = sae_mlpout.encode(mlp_post_act)
    return sae_mlpin_featmag
  
# %%
n_embd, n_feat = 2, 5
mlp = MLP(GPTConfig(n_embd=n_embd)).to(device)
sae_mlpin = DummySAE(n_embd, n_feat).to(device)
sae_mlpout = DummySAE(n_embd, n_feat).to(device)

# %%
activations = {}

def mlp_gelu_hook_fn(module, inputs, outputs):
    pre_actfn = inputs[0]
    post_actfn = outputs

    pre_actfn_copy = pre_actfn.detach().requires_grad_(True)

    with torch.enable_grad():
        recomputed_post_actfn = F.gelu(pre_actfn_copy, approximate="tanh")
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

# %%


batch, seq = 1, 1
sparse_feat_mags_in = torch.randn((batch, seq, n_feat), device=device)
# topk_indices_in = torch.arange(n_feat, device=device, dtype=torch.int64)
# topk_indices_in = einops.repeat(topk_indices_in, "k -> b s k", b=batch, s=seq)
sparse_feat_mags_out = sandwich(sparse_feat_mags_in, sae_mlpin, mlp, sae_mlpout)
hook_handle.remove() 
# %%
jacobian_exact = all_jacobian(
    sae_mlpin,
    sae_mlpout,
    mlp,
    activations["mlp_gelu"]
)

def compute_output_mags_fixed(s_feat_mags_in):
    return sandwich( # Use the version defined previously
        s_feat_mags_in,
        sae_mlpin,
        mlp,
        sae_mlpout
    )

sparse_feat_mags_in_grad = sparse_feat_mags_in.clone().detach().requires_grad_(True)

from torch.autograd.functional import jacobian

# Compute Jacobian using autograd
jacobian_auto_full = jacobian(
    func=compute_output_mags_fixed,
    inputs=sparse_feat_mags_in_grad,
    create_graph=False # Usually False unless you need higher-order derivatives
).squeeze()

print(jacobian_auto_full)
print(jacobian_exact)
torch.testing.assert_close(jacobian_auto_full.squeeze(), jacobian_exact.squeeze())
# )[0]



# %%

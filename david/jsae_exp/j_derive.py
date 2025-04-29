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

import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients()


config = GPTConfig(
    n_embd = 3,
)
k  =5
sae_config = SAEConfig(
    gpt_config = config,
    n_features = (10,),
    sae_variant = SAEVariant.TOPK,
    top_k = (k,)
)

mlp = MLP(config).to(device)
# sae_mlpin = TopKSAE(0, sae_config).to(device)
# sae_mlpout = TopKSAE(0, sae_config).to(device)

torch.manual_seed(0)
batch, seq = 1, 1
sparse_feat_mags_in = torch.randn((batch, seq, k), device=device)


class DummySAE(nn.Module):
    def __init__(self, n_embd: int, n_features: int):
        super().__init__()
        self.feature_size = n_features
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd)))
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features)))

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(x @ self.W_enc, k=k, dim=-1, sorted=True)
        mask = 
        
        top_k_values, top_k_indices = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()
        return latent_k_sparse
        
        
        #return x @ self.W_enc #torch.topk(x @ self.W_enc, k=k, dim=-1, sorted=True).values
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec


sae_mlpin = DummySAE(config.n_embd, sae_config.n_features[0]).to(device)
sae_mlpout = DummySAE(config.n_embd, sae_config.n_features[0]).to(device)


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
    sparse_feat_mags_out, topk_indices_out = torch.topk(sae_mlpin_featmag, k, dim=-1, sorted=True)
    return sparse_feat_mags_out, topk_indices_out
  
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



topk_indices_in = torch.arange(k, device=device, dtype=torch.int64)
topk_indices_in = einops.repeat(topk_indices_in, "k -> b s k", b=batch, s=seq)
sparse_feat_mags_out, topk_indices_out = sandwich(sparse_feat_mags_in, topk_indices_in)
hook_handle.remove() 
jacobian_exact = original_jacobian(
    sae_mlpin,
    sae_mlpout,
    mlp,
    topk_indices_in,
    topk_indices_out,
    activations["mlp_gelu"]
)

def sandwich_fixed_indices(
    sparse_feat_mags_in: Float[Tensor, "batch seq k_in"],
    topk_indices_in: Int[Tensor, "batch seq k_in"],
    fixed_topk_indices_out: Int[Tensor, "batch seq k_out"], # Use the indices found previously
    sae_mlpin, # : TopKSAE, # Use actual type hints if available
    sae_mlpout, # : TopKSAE,
    mlp, # : MLP
) -> Float[Tensor, "batch seq k_out"]:
    """
    Computes output feature magnitudes *at pre-determined indices*, making it differentiable w.r.t. input magnitudes.
    """
    # 1. Decode input features (using fixed input indices)
    sae_mlpin_featmag_full = untopk(sparse_feat_mags_in, topk_indices_in, sae_mlpin.feature_size)
    mlp_pre_act = sae_mlpin.decode(sae_mlpin_featmag_full)

    # 2. Pass through MLP
    mlp_post_act = mlp(mlp_pre_act) # Forward pass through MLP

    # 3. Encode with output SAE
    sae_mlpout_featmag_full = sae_mlpout.encode(mlp_post_act) # Full feature vector (batch, seq, feature_size)

    # 4. Select the magnitudes at the *fixed* output indices using gather
    # Ensure indices have the same number of dims as the tensor being gathered from
    sparse_feat_mags_out = torch.gather(sae_mlpout_featmag_full, dim=-1, index=fixed_topk_indices_out)

    return sparse_feat_mags_out

def compute_output_mags_fixed(s_feat_mags_in):
    return sandwich_fixed_indices( # Use the version defined previously
        s_feat_mags_in,
        topk_indices_in,        # Use fixed input indices
        topk_indices_out,       # Use fixed output indices
        sae_mlpin,
        sae_mlpout,
        mlp
    )

sparse_feat_mags_in_grad = sparse_feat_mags_in.clone().detach().requires_grad_(True)

from torch.autograd.functional import jacobian

# Compute Jacobian using autograd
jacobian_auto_full = jacobian(
    func=compute_output_mags_fixed,
    inputs=sparse_feat_mags_in_grad,
    create_graph=False # Usually False unless you need higher-order derivatives
)

# Extract the relevant block for comparison
# jacobian_auto_full shape: (batch, seq, k_out, batch, seq, k_in)
# We want the part where input (b, s) matches output (b, s)
# For batch=1, seq=1, this is jacobian_auto_full[0, 0, :, 0, 0, :]
jacobian_auto_compared = jacobian_auto_full[0, 0, :, 0, 0, :]

# Compare the manually computed Jacobian with the extracted autograd Jacobian
print("Jacobian Exact Shape:", jacobian_exact.shape)
print("Jacobian Auto (Full) Shape:", jacobian_auto_full.shape)
print("Jacobian Auto (Compared Slice) Shape:", jacobian_auto_compared.shape)

# Now compare jacobian_exact[0, 0] and jacobian_auto_compared
print("Comparison (should be close):")
print("Exact:", jacobian_exact[0, 0])
print("Auto:", jacobian_auto_compared)

# Check if they are close
are_close = torch.allclose(jacobian_exact[0, 0], jacobian_auto_compared, atol=1e-5)
print("Are they close?", are_close)


# jacobian_auto = torch.autograd.grad(
#     outputs = sparse_feat_mags_out,
#     inputs = sparse_feat_mags_in,
#     grad_outputs = torch.eye(5),
#     create_graph = True,
#     retain_graph = True
# )[0]



# %%

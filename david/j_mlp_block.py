
# %%
import os
# Change current working directory to parent
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())
# %%
import torch
from models.gpt import MLP
from models.sae.topk import TopKSAE
from models.gpt import GPTConfig

from pathlib import Path
from torch.nn import functional as F
import einops
from eindex import eindex
from jaxtyping import Float, Int
from torch import Tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn as nn



class DummySAE(nn.Module):
    def __init__(self, n_embd: int, n_features: int, k: int = 1):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn((n_features, n_embd)))
        self.W_enc = nn.Parameter(torch.randn((n_embd, n_features)))
        self.b_enc = nn.Parameter(torch.randn((n_features)))
        self.b_dec = nn.Parameter(torch.randn((n_embd)))
        self.k = k

    def encode(self, x: torch.Tensor, return_indicies = False) -> torch.Tensor:
        latent = (x - self.b_dec) @ self.W_enc + self.b_enc

        # Zero out all but the top-k activations
        top_k_values, top_k_indices = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()

        if return_indicies:
            return latent_k_sparse, top_k_indices
        else:
            return latent_k_sparse
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec

def get_jacobian_mlp_block(
    sae_mlpin,
    sae_mlpout,
    mlp: MLP,
    topk_indices_mlpin: Int[Tensor, "batch seq k"],
    topk_indices_mlpout: Int[Tensor, "batch seq k"],
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"], #d_mlp = 4 * n_embd
    norm_act_grads: Float[Tensor, "batch seq n_embd"] = None,
) -> Float[Tensor, "batch seq k2 k1"]:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)

    # #wd1 = sae_mlpin.W_dec @ mlp.W_in.T #(feat_size, d_model) @ (d_model, d_mlp) -> (feat_size, d_mlp)
    # wd1 = einops.einsum(sae_mlpin.W_dec, norm_act_grads, mlp.W_in.T, "feat_size d_model, batch seq d_model -> batch seq feat_size d_mlp") 
    # # w2e = mlp.W_out.T @ sae_mlpout.W_enc #(d_mlp, d_model) @ (d_model, feat_size) -> (d_mlp, feat_size)
    # w2e = einops.einsum(mlp.W_out.T, sae_mlpout.W_enc, "d_mlp d_model, d_model feat_size -> d_mlp feat_size") # (d_mlp, feat_size)

    wd1 = einops.einsum(sae_mlpin.W_dec, 
                        norm_act_grads, 
                        mlp.W_in, 
                        "feat_size n_embd, batch seq n_embd, d_mlp n_embd -> batch seq feat_size d_mlp")
    
    w2e = einops.einsum(mlp.W_out, sae_mlpout.W_enc, "n_embd d_mlp, n_embd feat_size -> d_mlp feat_size")

    wd1_topk_indices = eindex(
        wd1, topk_indices_mlpin, "batch seq [batch seq k] d_mlp -> batch seq k d_mlp"
    )

    dtype = wd1.dtype
    jacobian_mlp_path = einops.einsum(
        wd1_topk_indices,
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_mlpout],
        "batch seq k1 d_mlp, batch seq d_mlp, d_mlp batch seq k2 -> batch seq k2 k1", # ... = batch seq
    )

  
    jacobian_skip_path = (sae_mlpin.W_dec @ sae_mlpout.W_enc).T #(feat_size, n_embd) @ (n_embd, feat_size) -> (feat_size, feat_size)
    jacobian_skip_path = eindex(jacobian_skip_path, topk_indices_mlpout, topk_indices_mlpin,
                                "[batch seq k2] [batch seq k1] -> batch seq k2 k1")

    jacobian = jacobian_mlp_path + jacobian_skip_path
    return jacobian

def my_act_fn(x):
    return torch.tanh(x)

def my_act_fn_grad(input, output):
    # Gradient of tanh is 1 - tanh^2
    return 1 - output ** 2

activations = {}

def sandwich_mlp_block(in_feat_mags : Float[Tensor, "batch seq feat"],
             sae_mlpin : DummySAE,
             mlp : MLP,
             sae_mlpout : DummySAE) -> tuple[Float[Tensor, "batch seq feat"], Int[Tensor, "batch seq feat"]]:
    """
    takes sparse feature magnitudes and indices, and returns the sandwich product
    """
    
    top_k_values, in_feat_idx = torch.topk(in_feat_mags, sae_mlpin.k, dim=-1)
    mask = in_feat_mags >= top_k_values[..., -1].unsqueeze(-1)
    in_feat_mags_sparse = in_feat_mags * mask.float()

    resid_mid = sae_mlpin.decode(in_feat_mags_sparse)

    mlp_pre_act = my_act_fn(resid_mid)

    activations["norm_act_grad"] = my_act_fn_grad(resid_mid, mlp_pre_act)

    mlp_post_act = mlp(mlp_pre_act)

    resid_post = mlp_post_act + resid_mid

    out_feat_mags, out_feat_idx = sae_mlpout.encode(resid_post, return_indicies=True)

    return out_feat_mags, in_feat_idx, out_feat_idx


  
# %
n_embd, n_feat, k = 7, 13, 3
batch, seq = 2, 5
mlp = MLP(GPTConfig(n_embd=n_embd)).to(device)
sae_mlpin = DummySAE(n_embd, n_feat, k).to(device)
sae_mlpout = DummySAE(n_embd, n_feat, k).to(device)

# %%


def mlp_gelu_hook_fn(module, inputs, outputs):
    pre_actfn = inputs[0]
    post_actfn = outputs

    pre_actfn_copy = pre_actfn.detach().requires_grad_(True)

    print(f"pre_actfn_copy: {pre_actfn_copy.shape}")
    print(f"post_actfn: {post_actfn.shape}")

    with torch.enable_grad():
        recomputed_post_actfn = F.gelu(pre_actfn_copy, approximate="tanh")
        grad_of_actfn = torch.autograd.grad(
            outputs=recomputed_post_actfn,
            inputs=pre_actfn_copy,
            grad_outputs=torch.ones_like(recomputed_post_actfn),
            retain_graph=False,
            create_graph=False
        )[0]
    print(f"grad_of_actfn: {grad_of_actfn.shape}")

    activations["mlp_gelu"] = grad_of_actfn
    return outputs


# Hook registration


# %%

torch.manual_seed(42)
sparse_feat_mags_in = torch.randn((batch, seq, n_feat), device=device)
hook_handle = mlp.gelu.register_forward_hook(mlp_gelu_hook_fn)

sparse_feat_mags_out, in_feat_idx, out_feat_idx = sandwich_mlp_block(sparse_feat_mags_in, sae_mlpin, mlp, sae_mlpout)
hook_handle.remove() 

def sandwich_block_for_autograd(input_features):
    # We are differentiating the *final sparse output magnitudes*
    # w.r.t the *initial dense input feature magnitudes*.
    sparse_output_mags, _, _ = sandwich_mlp_block(input_features, sae_mlpin, mlp, sae_mlpout)
    return sparse_output_mags


jacobian_lucy = get_jacobian_mlp_block(
    sae_mlpin,
    sae_mlpout,
    mlp,
    in_feat_idx,
    out_feat_idx,
    activations["mlp_gelu"],
    activations["norm_act_grad"],
)

import torch.autograd.functional as autograd_F

raw_jacobian_torch_mlp_block = autograd_F.jacobian(
    sandwich_block_for_autograd,
    sparse_feat_mags_in,
    create_graph=True,
    vectorize=False,
)

print(f'{jacobian_lucy.shape=}')
print(f'{raw_jacobian_torch_mlp_block.shape=}')

from eindex import eindex
raw_diag_jacobian_torch_mlp_block = einops.einsum(raw_jacobian_torch_mlp_block, "b s k2 b s k1 -> b s k2 k1") #take diagonal along batch and seq
jacobian_torch_mlp_path_block = eindex(raw_diag_jacobian_torch_mlp_block, out_feat_idx, in_feat_idx, "b s [b s k2] [b s k1] -> b s k2 k1")
#jacobian_torch[out_feat_idx.squeeze()][:, in_feat_idx.squeeze()]
print(f'{jacobian_torch_mlp_path_block.shape=}')
torch.testing.assert_close(jacobian_lucy, jacobian_torch_mlp_path_block, rtol=1e-4, atol=1e-4)
print("Results are close!")


# %%

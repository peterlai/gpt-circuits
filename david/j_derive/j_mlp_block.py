
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
from models.gpt import GPTConfig, DynamicTanh

from pathlib import Path
from torch.nn import functional as F
import einops
from eindex import eindex
from jaxtyping import Float, Int
from typing import Optional, Callable
from torch import Tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn as nn
from eindex import eindex

import time
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

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

@timing
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


@timing
def get_jacobian_mlp_block_fast(
    sae_residmid,
    sae_residpost,
    mlp: MLP,
    topk_indices_residmid: Int[Tensor, "batch seq k1"],
    topk_indices_residpost: Int[Tensor, "batch seq k2"],
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"], # d_mlp = 4 * n_embd
    norm_act_grads: Float[Tensor, "batch seq n_embd"], # Cannot be None for this path
) -> Float[Tensor, "batch seq k2 k1"]:

    # --- Input Shapes ---
    # sae_residmid.W_dec: (feat_size, n_embd)
    # norm_act_grads:  (batch, seq, n_embd)
    # mlp.W_in:        (d_mlp, n_embd)
    # mlp.W_out:       (n_embd, d_mlp)
    # sae_residpost.W_enc:(n_embd, feat_size)
    # mlp_act_grads:   (batch, seq, d_mlp)
    # topk_indices_residmid: (batch, seq, k1)
    # topk_indices_residpost:(batch, seq, k2)
    # Output:          (batch, seq, k2, k1)

    if norm_act_grads is None:
         raise ValueError("norm_act_grads cannot be None for this Jacobian calculation path.")

    # --- MLP Path Calculation ---

    # 1. Index W_dec according to topk_indices_residmid
    # Original W_dec shape: (feat_size, n_embd)
    # Indices shape: (batch, seq, k1) -> selecting along the first dimension (feat_size)
    W_dec_indexed: Float[Tensor, "batch seq k1 n_embd"] = eindex(
        sae_residmid.W_dec, topk_indices_residmid, "[batch seq k1] n_embd -> batch seq k1 n_embd"
    )

    # 2. Compute the first part of the MLP path Jacobian contribution (related to input -> mlp pre-act)
    # Corresponds to: sum_n (W_dec_indexed[b, s, k1, n] * norm_act_grads[b, s, n] * W_in[d, n])
    # Avoids creating the large (b, s, feat_size, d_mlp) tensor.
    # We can perform this using matmul after element-wise multiplication.
    # term1 = W_dec_indexed * norm_act_grads.unsqueeze(2) # Shape: (b, s, k1, n)
    # wd1_topk = torch.matmul(term1, mlp.W_in.T) # Shapes: (b, s, k1, n) @ (n, d) -> (b, s, k1, d)
    # Let's try the 3-tensor einsum again, but on smaller tensors, potentially faster if optimized well.
    wd1_topk: Float[Tensor, "batch seq k1 d_mlp"] = einops.einsum(
        W_dec_indexed,
        norm_act_grads,
        mlp.W_in,
        "batch seq k1 n_embd, batch seq n_embd, d_mlp n_embd -> batch seq k1 d_mlp"
        # Alternative using matmul:
        # (W_dec_indexed * norm_act_grads.unsqueeze(2)) @ mlp.W_in.T
    )
    # wd1_topk shape: (batch, seq, k1, d_mlp)

    # 3. Compute the second part of the MLP path (related to mlp output act -> sae output features)
    # w2e = mlp.W_out.T @ sae_residpost.W_enc
    # mlp.W_out shape: (n, d), mlp.W_out.T shape: (d, n)
    # sae_residpost.W_enc shape: (n, f)
    # Result shape: (d, f)
    w2e: Float[Tensor, "d_mlp feat_size"] = torch.matmul(mlp.W_out.T, sae_residpost.W_enc)

    # 4. Index w2e according to topk_indices_residpost
    # Original w2e shape: (d_mlp, feat_size)
    # Indices shape: (batch, seq, k2) -> selecting along the second dimension (feat_size)
    w2e_indexed: Float[Tensor, "d_mlp batch seq k2"] = eindex(
        w2e, topk_indices_residpost, "d_mlp [batch seq k2] -> d_mlp batch seq k2"
    )
    # Permute for matmul: (batch, seq, d_mlp, k2)
    w2e_indexed_perm: Float[Tensor, "batch seq d_mlp k2"] = w2e_indexed.permute(1, 2, 0, 3)

    # 5. Combine parts for the MLP path Jacobian
    # Original einsum: "b s k1 d, b s d, d b s k2 -> b s k2 k1"
    # Optimized approach:
    # tmp = wd1_topk * mlp_act_grads.to(wd1_topk.dtype).unsqueeze(2) # Shape (b, s, k1, d)
    # jacobian_mlp_path_raw = torch.matmul(tmp, w2e_indexed_perm) # Shape (b, s, k1, k2)
    # jacobian_mlp_path = jacobian_mlp_path_raw.transpose(-1, -2) # Shape (b, s, k2, k1)

    # Perform step 5 efficiently:
    jacobian_mlp_path: Float[Tensor, "batch seq k2 k1"] = einops.einsum(
         wd1_topk,                                  # (b, s, k1, d)
         mlp_act_grads.to(wd1_topk.dtype),          # (b, s, d)
         w2e_indexed,                               # (d, b, s, k2)
        "batch seq k1 d_mlp, batch seq d_mlp, d_mlp batch seq k2 -> batch seq k2 k1"
        # This einsum combines the element-wise multiplication and the contraction.
        # Depending on the `opt_einsum` backend, this might be faster or slower than
        # the manual breakdown with matmul. Benchmarking is key.
    )

    # --- Skip Path Calculation ---

    # 1. Calculate the raw skip matrix M = W_dec @ W_enc
    # sae_residmid.W_dec shape: (f, n)
    # sae_residpost.W_enc shape: (n, f)
    # Result shape: (f, f)
    skip_matrix: Float[Tensor, "feat_size feat_size"] = torch.matmul(sae_residmid.W_dec, sae_residpost.W_enc)

    # 2. Index the skip matrix M[idx_in, idx_out]
    # Original code did eindex(M.T, idx_out, idx_in) -> M.T[idx_out, idx_in] -> M[idx_in, idx_out]
    # We compute M[idx_in, idx_out] directly and then transpose the k1, k2 dims.
    jacobian_skip_path_raw: Float[Tensor, "batch seq k1 k2"] = eindex(
        skip_matrix,
        topk_indices_residmid,   # Indexes first dim (rows) -> corresponds to input features
        topk_indices_residpost,  # Indexes second dim (cols) -> corresponds to output features
        "[batch seq k1] [batch seq k2] -> batch seq k1 k2"
    )

    # 3. Transpose the k1 and k2 dimensions to match the target shape (b, s, k2, k1)
    jacobian_skip_path: Float[Tensor, "batch seq k2 k1"] = jacobian_skip_path_raw.transpose(-1, -2)

    # --- Combine Paths ---
    jacobian = jacobian_mlp_path + jacobian_skip_path

    return jacobian




activations = {}

def get_elementwise_derivative(func: Callable, input : Tensor, output: Optional[Tensor] = None) -> Tensor: # (definition unchanged)
    with torch.enable_grad():
        if output is None:
            output = func(input)
        grads = torch.autograd.grad(outputs=output, inputs=input,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=False, retain_graph=True)[0]
    return grads

def sandwich_mlp_block(in_feat_mags : Float[Tensor, "batch seq feat"],
             sae_mlpin : DummySAE,
             mlp : MLP,
             sae_mlpout : DummySAE,
             dyt: DynamicTanh) -> tuple[Float[Tensor, "batch seq feat"], Int[Tensor, "batch seq feat"]]:
    """
    takes sparse feature magnitudes and indices, and returns the sandwich product
    """
    
    top_k_values, in_feat_idx = torch.topk(in_feat_mags, sae_mlpin.k, dim=-1)
    mask = in_feat_mags >= top_k_values[..., -1].unsqueeze(-1)
    in_feat_mags_sparse = in_feat_mags * mask.float()

    resid_mid = sae_mlpin.decode(in_feat_mags_sparse)

    mlp_pre_act = dyt(resid_mid)

    activations["norm_act_grad"] = get_elementwise_derivative(dyt, resid_mid, mlp_pre_act)

    mlp_post_act = mlp(mlp_pre_act)

    resid_post = mlp_post_act + resid_mid

    out_feat_mags, out_feat_idx = sae_mlpout.encode(resid_post, return_indicies=True)

    return out_feat_mags, in_feat_idx, out_feat_idx


  
# %
n_embd, n_feat, k = 128, 512, 10
batch, seq = 128,64
mlp = MLP(GPTConfig(n_embd=n_embd)).to(device)
sae_mlpin = DummySAE(n_embd, n_feat, k).to(device)
sae_mlpout = DummySAE(n_embd, n_feat, k).to(device)
dyt = DynamicTanh(n_embd).to(device)
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

sparse_feat_mags_out, in_feat_idx, out_feat_idx = sandwich_mlp_block(sparse_feat_mags_in, sae_mlpin, mlp, sae_mlpout, dyt)
hook_handle.remove() 

def sandwich_block_for_autograd(input_features):
    # We are differentiating the *final sparse output magnitudes*
    # w.r.t the *initial dense input feature magnitudes*.
    sparse_output_mags, _, _ = sandwich_mlp_block(input_features, sae_mlpin, mlp, sae_mlpout, dyt)
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

jacobian_fast = get_jacobian_mlp_block_fast(
    sae_mlpin,
    sae_mlpout,
    mlp,
    in_feat_idx,
    out_feat_idx,
    activations["mlp_gelu"],
    activations["norm_act_grad"],
)

# jacobian_fast_optimized_v2 = get_jacobian_mlp_block_fast_optimized_v2(
#     sae_mlpin,
#     sae_mlpout,
#     mlp,
#     in_feat_idx,
#     out_feat_idx,
#     activations["mlp_gelu"],
#     activations["norm_act_grad"],
# )

# import torch.autograd.functional as autograd_F

# @timing
# def compute_jacobian():
#     jacob = autograd_F.jacobian(
#         sandwich_block_for_autograd,
#         sparse_feat_mags_in,
#         create_graph=True,
#         vectorize=False,
#     )
    
#     jacob_diag = einops.einsum(jacob, "b s k2 b s k1 -> b s k2 k1") #take diagonal along batch and seq
#     return eindex(jacob_diag, out_feat_idx, in_feat_idx, "b s [b s k2] [b s k1] -> b s k2 k1")
    

# raw_jacobian_torch_mlp_block = compute_jacobian()

# print(f'{jacobian_lucy.shape=}')
# print(f'{raw_jacobian_torch_mlp_block.shape=}')

torch.testing.assert_close(jacobian_lucy, jacobian_fast, rtol=1e-4, atol=1e-4)
#torch.testing.assert_close(jacobian_fast, jacobian_fast_optimized_v2, rtol=1e-4, atol=1e-4)
print("Results are close!")



# %%

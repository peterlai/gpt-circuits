from jaxtyping import Float, Int
from typing import Optional
from torch import Tensor

from eindex import eindex
import einops
from models.gpt import MLP
import torch
from models.sae import SparseAutoencoder

@torch.compile(mode="max-autotune", fullgraph=True)
def jacobian_mlp(
    sae_mlpin : SparseAutoencoder,
    sae_mlpout : SparseAutoencoder,
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
    k = sae_mlpin.k
    jacobian = einops.einsum(
        wd1[topk_indices_mlpin],
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_mlpout],
        # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
        # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    ).abs_().sum() / (k ** 2)
    return jacobian

@torch.compile(mode="max-autotune", fullgraph=True)
def jacobian_mlp_block(
    sae_residmid,
    sae_residpost,
    mlp: MLP,
    topk_indices_residmid: Int[Tensor, "batch seq k"],
    topk_indices_residpost: Int[Tensor, "batch seq k"],
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"], #d_mlp = 4 * n_embd
    norm_act_grads: Float[Tensor, "batch seq n_embd"] = None,
) -> Float[Tensor, "batch seq k2 k1"]:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)

    wd1 = einops.einsum(sae_residmid.W_dec, 
                        norm_act_grads, 
                        mlp.W_in, 
                        "feat_size n_embd, batch seq n_embd, d_mlp n_embd -> batch seq feat_size d_mlp") #O(batch * seq * feat_size * d_mlp * n_embd) (!!)
    
    w2e = einops.einsum(mlp.W_out, sae_residpost.W_enc, "n_embd d_mlp, n_embd feat_size -> d_mlp feat_size") #O(n_embd * d_mlp * feat_size)

    wd1_topk_indices = eindex(
        wd1, topk_indices_residmid, "batch seq [batch seq k] d_mlp -> batch seq k d_mlp"
    )
    k = sae_residmid.k
    dtype = wd1.dtype
    jacobian_mlp_path = einops.einsum(
        wd1_topk_indices,
        mlp_act_grads.to(dtype),
        w2e[:, topk_indices_residpost],
        "batch seq k1 d_mlp, batch seq d_mlp, d_mlp batch seq k2 -> batch seq k2 k1", # ... = batch seq
    ) #O(batch * seq * k**2 * d_mlp)

  
    jacobian_skip_path = (sae_residmid.W_dec @ sae_residpost.W_enc).T #(feat_size, n_embd) @ (n_embd, feat_size) -> (feat_size, feat_size)
    jacobian_skip_path = eindex(jacobian_skip_path, topk_indices_residpost, topk_indices_residmid,
                                "[batch seq k2] [batch seq k1] -> batch seq k2 k1")

    jacobian = jacobian_mlp_path + jacobian_skip_path
    return jacobian.abs_().sum() / (k ** 2)

@torch.compile(mode="max-autotune", fullgraph=True)
def jacobian_mlp_block_fast(
    sae_mlpin,
    sae_mlpout,
    mlp: MLP,
    topk_indices_mlpin: Int[Tensor, "batch seq k1"],
    topk_indices_mlpout: Int[Tensor, "batch seq k2"],
    mlp_act_grads: Float[Tensor, "batch seq d_mlp"], # d_mlp = 4 * n_embd
    norm_act_grads: Float[Tensor, "batch seq n_embd"], # Cannot be None for this path
) -> Float[Tensor, "batch seq k2 k1"]:

    # --- Input Shapes ---
    # sae_mlpin.W_dec: (feat_size, n_embd)
    # norm_act_grads:  (batch, seq, n_embd)
    # mlp.W_in:        (d_mlp, n_embd)
    # mlp.W_out:       (n_embd, d_mlp)
    # sae_mlpout.W_enc:(n_embd, feat_size)
    # mlp_act_grads:   (batch, seq, d_mlp)
    # topk_indices_mlpin: (batch, seq, k1)
    # topk_indices_mlpout:(batch, seq, k2)
    # Output:          (batch, seq, k2, k1)

    if norm_act_grads is None:
         raise ValueError("norm_act_grads cannot be None for this Jacobian calculation path.")

    # --- MLP Path Calculation ---

    # 1. Index W_dec according to topk_indices_mlpin
    # Original W_dec shape: (feat_size, n_embd)
    # Indices shape: (batch, seq, k1) -> selecting along the first dimension (feat_size)
    W_dec_indexed: Float[Tensor, "batch seq k1 n_embd"] = eindex(
        sae_mlpin.W_dec, topk_indices_mlpin, "[batch seq k1] n_embd -> batch seq k1 n_embd"
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
    # w2e = mlp.W_out.T @ sae_mlpout.W_enc
    # mlp.W_out shape: (n, d), mlp.W_out.T shape: (d, n)
    # sae_mlpout.W_enc shape: (n, f)
    # Result shape: (d, f)
    w2e: Float[Tensor, "d_mlp feat_size"] = torch.matmul(mlp.W_out.T, sae_mlpout.W_enc)

    # 4. Index w2e according to topk_indices_mlpout
    # Original w2e shape: (d_mlp, feat_size)
    # Indices shape: (batch, seq, k2) -> selecting along the second dimension (feat_size)
    w2e_indexed: Float[Tensor, "d_mlp batch seq k2"] = eindex(
        w2e, topk_indices_mlpout, "d_mlp [batch seq k2] -> d_mlp batch seq k2"
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
    # sae_mlpin.W_dec shape: (f, n)
    # sae_mlpout.W_enc shape: (n, f)
    # Result shape: (f, f)
    skip_matrix: Float[Tensor, "feat_size feat_size"] = torch.matmul(sae_mlpin.W_dec, sae_mlpout.W_enc)

    # 2. Index the skip matrix M[idx_in, idx_out]
    # Original code did eindex(M.T, idx_out, idx_in) -> M.T[idx_out, idx_in] -> M[idx_in, idx_out]
    # We compute M[idx_in, idx_out] directly and then transpose the k1, k2 dims.
    jacobian_skip_path_raw: Float[Tensor, "batch seq k1 k2"] = eindex(
        skip_matrix,
        topk_indices_mlpin,   # Indexes first dim (rows) -> corresponds to input features
        topk_indices_mlpout,  # Indexes second dim (cols) -> corresponds to output features
        "[batch seq k1] [batch seq k2] -> batch seq k1 k2"
    )

    # 3. Transpose the k1 and k2 dimensions to match the target shape (b, s, k2, k1)
    jacobian_skip_path: Float[Tensor, "batch seq k2 k1"] = jacobian_skip_path_raw.transpose(-1, -2)

    # --- Combine Paths ---
    jacobian = jacobian_mlp_path + jacobian_skip_path

    k = sae_mlpin.k

    return jacobian.abs_().sum() / (k ** 2)

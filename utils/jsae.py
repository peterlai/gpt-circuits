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
    k = sae_mlpin.k
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
    return jacobian.abs_().sum() / (k ** 2)

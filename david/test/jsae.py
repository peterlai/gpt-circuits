# %%
%load_ext autoreload
%autoreload 2
# %%
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Change current working directory to parent
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

import torch
from pathlib import Path
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import StaircaseTopKSAE
from models.gpt import GPT
from models.jsaesparsified import JSparsifiedGPT
from data.tokenizers import ASCIITokenizer
from david.convert_to_tl import convert_gpt_to_transformer_lens
from david.convert_to_tl import run_tests as run_tl_tests
from transformer_lens.hook_points import HookPoint
from models.sae import SparseAutoencoder
from models.gpt import MLP

from typing import Tuple
import einops

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients(sparsity = (1,1,1,1))

# gpt_dir = Path("checkpoints/shakespeare_64x4")
# gpt = GPT.load(gpt_dir, device=device)

# sae_config =SAEConfig(
#         gpt_config=gpt.config,
#         n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
#         sae_variant=SAEVariant.TOPK,
#         top_k = (10, 10, 10, 10, 10, 10, 10, 10)
#     )

gpt_mlp = JSparsifiedGPT.load("checkpoints/mlp-topk.shakespeare_64x4", 
                                 loss_coefficients,
                                 trainable_layers = None,
                                 device = device)
gpt_mlp.to(device)
#tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, sae_config)

prompt = "Second Servingman:\nI will not so"
tokenizer = ASCIITokenizer()
tokens = tokenizer.encode(prompt)
tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)

x = tokens[:, :-1].contiguous()
y = tokens[:, 1:].contiguous()
out = gpt_mlp(x, targets=y, is_eval=False)
# %%
tl_gpt_mlp = convert_gpt_to_transformer_lens(gpt_mlp.gpt, gpt_mlp.config)

tl_logits, cache = tl_gpt_mlp.run_with_cache(x)


def get_jacobian(
    sae_pair: Tuple[SparseAutoencoder, SparseAutoencoder],
    mlp: MLP,
    topk_indices: torch.Tensor,
    mlp_act_grads: torch.Tensor,
    topk_indices2: torch.Tensor,
) -> torch.Tensor:
    wd1 = sae_pair[0].W_dec @ mlp.W_in.T
    w2e = mlp.W_out.T @ sae_pair[1].W_enc

    jacobian = einops.einsum(
        wd1[topk_indices],
        mlp_act_grads,
        w2e[:, topk_indices2],
        # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
        # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    )

    return jacobian


def run_sandwich(
    sae_pair: Tuple[SparseAutoencoder, SparseAutoencoder],
    mlp_with_act_grads: MLP,
    ln_out_act: torch.Tensor,
    use_recontr_mlp_input: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sae_acts1, topk_indices1 = sae_pair[0].encode(
        ln_out_act, return_topk_indices=True
    )
    act_reconstr = sae_pair[0].decode(sae_acts1, False)
    mlp_out, mlp_act_grads = mlp_with_act_grads(
        act_reconstr if use_recontr_mlp_input else ln_out_act,
        return_act_grads=True
    )
    sae_acts2, topk_indices2 = sae_pair[1].encode(mlp_out, return_topk_indices=True)

    jacobian = get_jacobian(
        sae_pair, mlp_with_act_grads, topk_indices1, mlp_act_grads, topk_indices2
    )

    acts_dict = {
        "sae_acts1": sae_acts1,
        "topk_indices1": topk_indices1,
        "act_reconstr": act_reconstr,
        "mlp_out": mlp_out,
        "mlp_act_grads": mlp_act_grads,
        "sae_acts2": sae_acts2,
        "topk_indices2": topk_indices2,
    }

    return jacobian, acts_dict

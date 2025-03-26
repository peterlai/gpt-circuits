# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os

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
from data.tokenizers import ASCIITokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_coefficients = LossCoefficients()

def load_gpt_model():
    gpt_dir = Path("checkpoints/shakespeare_64x4")
    gpt = GPT.load(gpt_dir, device=device)
    tokenizer = ASCIITokenizer()
    return gpt, tokenizer

gpt, tokenizer = load_gpt_model()


saes = StaircaseTopKSAE(
    layer_idx=0,
    config=SAEConfig(
        gpt_config=gpt.config,
        n_features=tuple(64 * n for n in (2, 4, 8)),
        sae_variant=SAEVariant.TOPK_STAIRCASE,
        top_k=(10,10,10),
    ),
    loss_coefficients=loss_coefficients,
    model=gpt,
)

os.makedirs("checkpoints/staircase_test", exist_ok=True)
saes.save(Path("checkpoints/staircase_test"))
# %%

new_saes = StaircaseTopKSAE(
    layer_idx=0,
    config=SAEConfig(
        gpt_config=gpt.config,
        n_features=tuple(64 * n for n in (2, 4, 8)),
        sae_variant=SAEVariant.TOPK_STAIRCASE,
        top_k=(10,10,10),
    ),
    loss_coefficients=loss_coefficients,
    model=gpt,
)

new_saes.load(Path("checkpoints/staircase_test"), device=device)
# %%
# Check if all parameters match and are the same
def compare_parameters(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} != {name2}")
            return False
        if not torch.equal(param1, param2):
            print(f"Parameter value mismatch for {name1}")
            return False
    return True

if compare_parameters(saes, new_saes):
    print("All parameters match and are the same.")
else:
    print("There are mismatches in the parameters.")
# %%
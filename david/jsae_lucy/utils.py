import os
from typing import Optional

import einops
from safetensors.torch import load_file
import torch
from transformer_lens import HookedTransformer
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

import wandb
from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.training.training_sae_pair import TrainingSAEPair
from jacobian_saes.training.mlp_with_act_grads import MLPWithActGrads
from jacobian_saes.sae_training_runner import RANDOMIZED_LLM_WEIGHTS_PATH

api = wandb.Api()
default_device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

default_prompt = "Given the existence as uttered forth in the public works of Puncher and Wattmann of a personal God quaquaquaqua with white beard quaquaquaqua outside time without extension who from the heights of divine apathia divine athambia divine aphasia loves us dearly with some exceptions for reasons unknown but time will tell and suffers like the divine Miranda with those who for reasons unknown but time will tell are plunged in torment plunged in fire whose fire flames if that continues and who can doubt it will fire the firmament that is to say blast hell to heaven so blue still and calm so calm with a calm which even though intermittent is better than nothing but not so fast and considering what is more that as a result of the labours left unfinished crowned by the Acacacacademy of Anthropopopometry of Essy-in-Possy of Testew and Cunard it is established beyond all doubt all other doubt than that which clings to the labours of men that as a result of the labours unfinished of Testew and Cunard it is established as hereinafter but not so fast for reasons unknown that as a result of the public works of Puncher and Wattmann it is established beyond all doubt that in view of the labours of Fartov and Belcher left unfinished for reasons unknown of Testew and Cunard left unfinished it is established what many deny that man in Possy of Testew and Cunard that man in Essy that man in short that man in brief in spite of the strides of alimentation and defecation is seen to waste and pine waste and pine and concurrently simultaneously what is more for reasons unknown in spite of the strides of physical culture the practice of sports such as tennis football running cycling swimming flying floating riding gliding conating camogie skating tennis of all kinds dying flying sports of all sorts autumn summer winter winter tennis of all kinds hockey of all sorts penicilline and succedanea in a word I resume and concurrently simultaneously for reasons unknown to shrink and dwindle in spite of the tennis I resume flying gliding golf over nine and eighteen holes tennis of all sorts in a word for reasons unknown in Feckham Peckham Fulham Clapham namely concurrently simultaneously what is more for reasons unknown but time will tell to shrink and dwindle I resume Fulham Clapham in a word the dead loss per caput since the death of Bishop Berkeley being to the tune of one inch four ounce per caput approximately by and large more or less to the nearest decimal good measure round figures stark naked in the stockinged feet in Connemara in a word for reasons unknown no matter what matter the facts are there and considering what is more much more grave that in the light of the labours lost of Steinweg and Peterman it appears what is more much more grave that in the light the light the light of the labours lost of Steinweg and Peterman that in the plains in the mountains by the seas by the rivers running water running fire the air is the same and then the earth namely the air and then the earth in the great cold the great dark the air and the earth abode of stones in the great cold alas alas in the year of their Lord six hundred and something the air the earth the sea the earth abode of stones in the great deeps the great cold an sea on land and in the air I resume for reasons unknown in spite of the tennis the facts are there but time will tell I resume alas alas on on in short in fine on on abode of stones who can doubt it I resume but not so fast I resume the skull to shrink and waste and concurrently simultaneously what is more for reasons unknown in spite of the tennis on on the beard the flames the tears the stones so blue so calm alas alas on on the skull the skull the skull the skull in Connemara in spite of the tennis the labours abandoned left unfinished graver still abode of stones in a word I resume alas alas abandoned unfinished the skull the skull in Connemara in spite of the tennis the skull alas the stones Cunard tennis... the stones... so calm... Cunard... unfinished..."


def load_pretrained(
    wandb_artifact_path: str, device: str = default_device,
    use_training_class: bool = False,
) -> tuple[SAEPair, HookedTransformer, MLPWithActGrads, int]:
    local_path = os.path.join("artifacts/", wandb_artifact_path.split("/")[-1])

    if not os.path.exists(local_path):
        artifact = api.artifact(wandb_artifact_path)
        artifact.download()


    sae_pair = SAEPair.load_from_pretrained(local_path, device=device)
    if getattr(sae_pair.cfg, "randomize_llm_weights", False):
        model = HookedTransformer.from_pretrained_no_processing(sae_pair.cfg.model_name,
                                                                device=sae_pair.device)
        llm_weights_path = os.path.join(local_path, RANDOMIZED_LLM_WEIGHTS_PATH)
        model.load_state_dict(load_file(llm_weights_path))
    else:
        model = HookedTransformer.from_pretrained(sae_pair.cfg.model_name,
                                                  device=sae_pair.device)

    if use_training_class:
        sae_pair = TrainingSAEPair.load_from_pretrained(
            "artifacts/" + wandb_artifact_path.split("/")[-1],
            transformer_block=model.blocks[sae_pair.cfg.hook_layer],
            llm_cfg=model.cfg,
            device=device,
        )
    layer = sae_pair.cfg.hook_layer
    mlp = model.blocks[layer].mlp

    mlp_with_grads = MLPWithActGrads(mlp.cfg)
    mlp_with_grads.load_state_dict(mlp.state_dict())
    mlp_with_grads.to(sae_pair.cfg.device)

    return sae_pair, model, mlp_with_grads, layer


def get_jacobian(
    sae_pair: SAEPair,
    mlp: CanBeUsedAsMLP,
    topk_indices: torch.Tensor,
    mlp_act_grads: torch.Tensor,
    topk_indices2: torch.Tensor,
) -> torch.Tensor:
    wd1 = sae_pair.get_W_dec(False) @ mlp.W_in
    w2e = mlp.W_out @ sae_pair.get_W_enc(True)

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
    sae_pair: SAEPair,
    mlp_with_act_grads: MLPWithActGrads,
    ln_out_act: torch.Tensor,
    use_recontr_mlp_input: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sae_acts1, topk_indices1 = sae_pair.encode(
        ln_out_act, False, return_topk_indices=True
    )
    act_reconstr = sae_pair.decode(sae_acts1, False)
    mlp_out, mlp_act_grads = mlp_with_act_grads(
        act_reconstr if use_recontr_mlp_input else ln_out_act
    )
    sae_acts2, topk_indices2 = sae_pair.encode(mlp_out, True, return_topk_indices=True)

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

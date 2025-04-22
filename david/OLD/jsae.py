from typing import Optional

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import TopKSAE
from models.mlp import MLP
import torch
from jacobian_saes.utils import get_jacobian
import einops
import torch.nn as nn

@dataclass
class JSAEActivations:
    sae_acts1: torch.Tensor
    topk_indices1: torch.Tensor
    act_reconstr: torch.Tensor
    mlp_out: torch.Tensor
    mlp_act_grads: torch.Tensor

class JSAE(TopKSAE):
    """
    Jacobian Sparse Autoencoder.
    """

    def __init__(self, 
                 layer_idx: int,  # Added missing comma
                 config: SAEConfig, 
                 loss_coefficients: LossCoefficients,
                 model: nn.Module):
        super().__init__(layer_idx, config, loss_coefficients, model)
        suffix = 'mlpin' if layer_idx % 2 == 0 else 'mlpout'
        sae_idx = layer_idx // 2
        self.key = f"{sae_idx}_{suffix}"
        self.is_mlp_out = suffix == 'mlpout'
        
    def get_jacobian(
        self,
        topk_indices: torch.Tensor,
        mlp_act_grads: torch.Tensor,
        topk_indices2: torch.Tensor,
    ) -> torch.Tensor:
        wd1 = self.sae_mlpin.W_dec @ self.mlp.W_in
        w2e = self.mlp.W_out @ self.sae_mlpout.W_enc

        jacobian = einops.einsum(
            wd1[topk_indices],
            mlp_act_grads,
            w2e[:, topk_indices2],
            # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
            # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
            "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
        )

        return jacobian
    
    def forward(
        self,
        act: torch.Tensor,
        use_recontr_mlp_input: bool = False,
        prev_act: Optional[JSAEActivations] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        if not self.is_mlp_out:
            feature_magnitudes, topk_indices1 = self.encode(act, return_topk_indices=True)
            act_reconstr = self.decode(feature_magnitudes)
            
            
            
        if self.is_mlp_out:
            
            
            
            
        
        feature_magnitudes, topk_indices1 = self.encode(act, return_topk_indices=True)
        act_reconstr = self.decode(feature_magnitudes)
        
        mlp_out, mlp_act_grads = self.mlp(
            act_reconstr if use_recontr_mlp_input else act
        )
        
        sae_acts2, topk_indices2 = self.sae_mlpout.encode(mlp_out, return_topk_indices=True)

        jacobian = get_jacobian(
            topk_indices1, mlp_act_grads, topk_indices2
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
    
        def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (B, T, embedding size)
        """
        feature_magnitudes = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)
        if self.should_return_losses:
            sparsity_loss = torch.tensor(0.0, device=x.device)  # no need for sparsity loss for top-k SAE
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output
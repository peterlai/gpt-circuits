import torch
from jaxtyping import Float
from torch import nn
from transformer_lens.components.mlps.mlp import MLP as TransformerLensMLP
from transformer_lens.utilities.addmm import batch_addmm


class MLPWithActGrads(TransformerLensMLP):
    """
    A clone of the MLP in the LLM which also returns the gradients
    of the activation function.
    Necessary for computing the Jacobian loss.
    """
    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> tuple[
        Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos d_mlp"]
    ]:
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = self.hook_pre(
            batch_addmm(self.b_in, self.W_in, x)
        )  # [batch, pos, d_mlp]

        if (
            self.cfg.is_layer_norm_activation()
            and self.hook_mid is not None
            and self.ln is not None
            and not self.cfg.use_normalization_before_and_after
        ):
            raise NotImplementedError(
                "You passed in something weird and I can't be bothered to support it rn, go check out the TransformerLens MLP code for what's supposed to go here and open a PR if you want this to work"
            )
        else:
            with torch.enable_grad():
                if not pre_act.requires_grad:
                    pre_act.requires_grad = True
                post_act = self.act_fn(pre_act)  # [batch, pos, d_mlp]
                grad_of_act = torch.autograd.grad(
                    outputs=post_act, inputs=pre_act,
                    grad_outputs=torch.ones_like(post_act), retain_graph=True
                )[0]
            post_act = self.hook_post(post_act)
        output = batch_addmm(self.b_out, self.W_out, post_act)

        return output, grad_of_act

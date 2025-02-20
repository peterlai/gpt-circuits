from typing import Optional

import torch
from torch.optim import Optimizer

from config.sae.training import SAETrainingConfig
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training import Trainer
from training.gpt import GPTTrainer


class SAETrainer(Trainer):
    """
    Base class for sparsified GPT trainers.
    """

    config: SAETrainingConfig
    model: SparsifiedGPT

    # Checkpoint metrics once training is complete
    checkpoint_l0s: torch.Tensor
    checkpoint_ce_loss: torch.Tensor
    checkpoint_ce_loss_increases: torch.Tensor
    checkpoint_compound_ce_loss_increase: torch.Tensor

    def __init__(self, model: SparsifiedGPT, config: SAETrainingConfig):
        """
        Initialize the trainer.
        """

        self.checkpoint_l0s = torch.zeros((len(model.saes),), device=config.device)
        self.checkpoint_ce_loss = torch.tensor(float("inf"), device=config.device)
        self.checkpoint_ce_loss_increases = torch.zeros((len(model.saes),), device=config.device)
        self.checkpoint_compound_ce_loss_increase = torch.tensor(0.0, device=config.device)

        super().__init__(model, config)

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Calculate model loss.
        """
        output: SparsifiedGPTOutput = self.model(x, y, is_eval=is_eval)
        loss = self.output_to_loss(output)
        metrics = None

        # Only include metrics if in evaluation mode
        if is_eval:
            metrics = self.gather_metrics(loss, output)

        return loss, metrics

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Convert model output to loss.
        """
        ...

    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        # Add SAE metrics
        sae_l0s = torch.stack([loss_components.l0 for loss_components in output.sae_loss_components.values()])
        metrics = {
            "loss": loss,
            "ce_loss": output.cross_entropy_loss,
            "sae_losses": output.sae_losses,
            "ce_loss_increases": output.ce_loss_increases,
            "compound_ce_loss_increase": output.compound_ce_loss_increase,
            "l0s": sae_l0s,
        }

        # Add extra GPT metrics
        metrics.update(
            {
                "stream_l1s": torch.stack(
                    [sae_loss_components.x_l1 for sae_loss_components in output.sae_loss_components.values()]
                )
            }
        )

        return metrics

    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
        # Train weights.
        super().train()

        # Wait for all processes to complete training.
        if self.ddp:
            torch.distributed.barrier()

        # Reload all checkpoint weights, which may include those that weren't trained.
        self.model = SparsifiedGPT.load(
            self.config.out_dir,
            loss_coefficients=self.config.loss_coefficients,
            trainable_layers=None,  # Load all layers
            device=self.config.device,
        ).to(self.config.device)

        # Gather final metrics. We don't bother compiling because we're just running eval once.
        final_metrics = self.val_step(0, should_log=False)  # step 0 so checkpoint isn't saved.
        self.checkpoint_l0s = final_metrics["l0s"]
        self.checkpoint_ce_loss = final_metrics["ce_loss"]
        self.checkpoint_ce_loss_increases = final_metrics["ce_loss_increases"]
        self.checkpoint_compound_ce_loss_increase = final_metrics["compound_ce_loss_increase"]

        # Summarize results
        if self.is_main_process:
            print(f"Final L0s: {self.pretty_print(self.checkpoint_l0s)}")
            print(f"Final CE loss increases: {self.pretty_print(self.checkpoint_ce_loss_increases)}")
            print(f"Final compound CE loss increase: {self.pretty_print(self.checkpoint_compound_ce_loss_increase)}")

    def configure_optimizer(self, model: SparsifiedGPT) -> Optimizer:
        """
        Configure the optimizer for training a sparsified GPT model.
        """
        # Get existing param groups for GPT model.
        gpt_param_groups = GPTTrainer.get_param_groups(model.gpt, self.config)

        # Add SAE parameters to the optimizer.
        sae_params = [p for p in model.saes.parameters() if p.requires_grad]
        num_gpt_params = sum(p.numel() for g in gpt_param_groups for p in g["params"])
        num_sae_params = sum(p.numel() for p in sae_params)

        # Print number of parameters
        if self.is_main_process:
            print(f"Trainable GPT parameters: {num_gpt_params:,}")
            print(f"Trainable SAE parameters: {num_sae_params:,}")

        # We set weight_decay to 0.0 for SAE parameters.
        param_groups = gpt_param_groups + [{"params": sae_params, "weight_decay": 0.0}]

        # Create optimizer
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.is_fused_adamW_available,
        )

from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel
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
        # TODO: This is a hack as a result of the way the loss is computed for the JSAETrainer
        if "jsae" in self.config.sae_config.sae_variant:
            split_idx = 2 * self.model.config.gpt_config.n_layer
            sae_loss, jacobian_loss = loss[:split_idx], loss[split_idx:]
        else:
            sae_loss = loss
    
        
        
        sae_l0s = torch.stack([loss_components.l0 for loss_components in output.sae_loss_components.values()])
        metrics = {
            "loss": sae_loss,
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
        
        if "staircase" in self.config.sae_config.sae_variant:
            l0_per_chunk = {}
            for layer_idx, feature_magnitudes in output.feature_magnitudes.items():
                num_chunks = layer_idx + 1
                grouped_feature_magnitudes = torch.chunk(feature_magnitudes, num_chunks, dim=-1) # tuple[(batch, seq, feature_size_each_chunk)]
                grouped_feature_magnitudes = torch.stack(grouped_feature_magnitudes, dim=-2) # (batch, seq, n_chunks, feature_size_each_chunk)
                grouped_l0 = (grouped_feature_magnitudes != 0).float().sum(dim=-1) # (batch, seq, n_chunks)
                l0_per_chunk[layer_idx] = grouped_l0.mean(dim=(0,1)) # (n_chunks)
                metrics[f"l0_{layer_idx}"] = l0_per_chunk[layer_idx]
                
        if "jsae" in self.config.sae_config.sae_variant:
            metrics["∇_l1"] = jacobian_loss
            
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
        # NOTE: We're using `model_type` to account for use of subclasses.
        self.model = self.model_type.load(
            self.config.out_dir,
            loss_coefficients=self.config.loss_coefficients,
            trainable_layers=None,  # Load all layers
            device=self.config.device,
        ).to(self.config.device)

        # Wrap the model if using DDP
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])  # type: ignore

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
        sae_params = [param for (key, param) in model.named_parameters() if param.requires_grad and key.split(".")[0] != "gpt"]
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

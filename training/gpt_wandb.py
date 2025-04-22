"""
Train GPT model:
$ python -m training.gpt --config=shakespeare_64x4

DDP launch for e.g. 8 GPUs:
$ torchrun --standalone --nproc_per_node=8 -m training.gpt --config=stories_256x4
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

import wandb
from config import TrainingConfig
from config.gpt.training import GPTTrainingConfig, options
from models.gpt import GPT, NormalizationStrategy
from training import Trainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    parser.add_argument("--norm", type=str, help="Normalization strategy [LayerNorm, DyanmicTanh,Identity]")
    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="gpt-sweep", help="Wandb project name")
    parser.add_argument('--max_steps', type=int, default=10000, help='Total number of training steps')
    parser.add_argument('--lr_end', type=float, default=1e-5, help='Final learning rate')
    parser.add_argument('--alpha_mlp', type=float, default=10, help='Alpha for mlp')

    return parser.parse_args()


class GPTTrainer(Trainer):
    """
    Trainer for GPT models.
    """

    def __init__(self, config: GPTTrainingConfig):
        """
        Load new GPT model from config.
        """
        model = GPT(config.gpt_config)

        super().__init__(model, config)

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Calculate model loss.
        """
        _, loss = self.model(x, y)

        return loss, None

    def configure_optimizer(self, model: GPT) -> Optimizer:
        """
        Configure the optimizer for training a GPT model.
        """
        # Get parameter groups for GPT model.
        param_groups = self.get_param_groups(model, self.config, verbose=self.is_main_process)

        # Create optimizer
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.is_fused_adamW_available,
        )

    @classmethod
    def get_param_groups(cls, model: GPT, config: TrainingConfig, verbose: bool = False) -> list[dict]:
        """
        Get parameter groups for the model.
        """
        # Start with all of the candidate parameters (that require grad).
        param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Print number of parameters in each group.
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        return optim_groups

    @torch.no_grad()
    def val_step(self, step, should_log=True) -> dict[str, torch.Tensor]:
        """
        Perform one step of validation.
        """
        self.model.eval()
        self.val_dataloader.reset()
        loss_accum = torch.tensor(0.0, device=self.device)
        metrics_accum: dict[str, torch.Tensor] = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        for _ in range(self.eval_steps):
            x, y = self.val_dataloader.next_batch(self.device)
            with torch.autocast(device_type=self.autocast_device_type, dtype=torch.bfloat16):
                loss, metrics = self.calculate_loss(x, y, is_eval=True)

            # Accumulate loss
            loss_accum = loss_accum + loss / self.eval_steps

            # Accumulate metrics
            metrics = metrics or {}
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum[k] + v / self.eval_steps

        if self.ddp:
            distributed.all_reduce(loss_accum, op=distributed.ReduceOp.AVG)

            # TODO: Does this work?
            for k, v in metrics_accum.items():
                distributed.all_reduce(v, op=distributed.ReduceOp.AVG)

        if self.is_main_process:
            # Save the model if it's the best we've seen so far
            best_val_loss = torch.min(self.best_val_loss, loss_accum)
            is_best = best_val_loss == loss_accum
            # We're using a quirky comparison that allows `loss` to have dimensionality.
            if self.best_val_loss.tolist() != best_val_loss.tolist() and step > 0:
                self.best_val_loss = best_val_loss
                self.save_checkpoint(self.unwrapped_model, is_best)

            # Log metrics unless skipped
            if should_log:
                log_data = {
                    "type": "eval",
                    "step": step,
                    "loss": loss_accum,
                    "checkpoint": is_best if step > 0 else False,
                    **metrics_accum,
                }
                self.log(log_data, self.LogDestination.EVAL)

                # Log to wandb
                wandb_log_data = {
                    "eval/loss": loss_accum.item(),
                    "eval/best_loss": self.best_val_loss.item(),
                    **{f"eval/{k}": v.item() for k, v in metrics_accum.items()},
                }
                wandb.log(wandb_log_data, step=step)

        return metrics_accum

    def train_step(self, step):
        """
        Perform one step of training optimization.
        """
        t0 = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=self.device)
        for micro_step in range(self.gradient_accumulation_steps):
            x, y = self.train_dataloader.next_batch(self.device)
            x, y = x.to(self.device), y.to(self.device)
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = micro_step == self.gradient_accumulation_steps - 1  # type: ignore
            with torch.autocast(device_type=self.autocast_device_type, dtype=torch.bfloat16):
                loss, _ = self.calculate_loss(x, y, is_eval=False)

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / self.gradient_accumulation_steps
            loss_accum = loss_accum + loss.detach()

            self.backward(loss)

        if self.ddp:
            distributed.all_reduce(loss_accum, op=distributed.ReduceOp.AVG)

        # clip the gradients (if a grad clip value is provided)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip or float("inf"))

        # determine and set the learning rate for this iteration
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds

        if self.is_main_process and step % self.config.log_interval == 0:
            log_data = {
                "type": "train",
                "step": step,
                "loss": loss_accum,
                "lr": f"{lr:.1e}",
                "norm": norm,
                "dt": f"{dt:.3f}",
            }
            self.log(log_data, self.LogDestination.TRAIN)

            # Log to wandb
            wandb_log_data = {
                "train/loss": loss_accum.item(),
                "train/lr": lr,
                "train/grad_norm": norm.item(),
                "train/dt_ms": dt * 1000, # Log time in ms
            }
            wandb.log(wandb_log_data, step=step)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Initialize wandb
    # Hyperparameters from the sweep will be in wandb.config
    # We pass the base config and args to wandb.init to log them
    # wandb.config will automatically merge sweep params, cmd args, and config file values
    run = wandb.init(
        project=args.wandb_project,
        config=vars(args), # Log command line args
        mode = 'disabled'
    )
    # Update config with hyperparameters from wandb sweep
    # Important: wandb.config contains the merged config (sweep > cmd > file)
    # config.max_steps = wandb.config.max_steps if hasattr(wandb.config, 'max_steps') else config.max_steps
    # config.min_lr = wandb.config.lr_end if hasattr(wandb.config, 'lr_end') else config.min_lr
    # config.gpt_config.alpha_mlp = wandb.config.alpha_mlp if hasattr(wandb.config, 'alpha_mlp') else config.gpt_config.alpha_mlp

    # Update outdir if name is provided
    if args.name:
        config.name = args.name
    # Use wandb run directory for output to avoid conflicts between sweep runs
    # wandb.run.dir provides a unique directory for each run

    if args.norm:
        config.gpt_config.norm_strategy = args.norm

    if args.alpha_mlp:
        config.gpt_config.alpha_mlp = args.alpha_mlp

    if args.lr_end:
        config.min_lr = args.lr_end

    if args.max_steps:
        config.max_steps = args.max_steps

    # Initialize trainer
    trainer = GPTTrainer(config)
    for i in range(trainer.model.config.n_layer):
        print(f"Layer {i} attn has norm strategy {trainer.model.transformer.h[i].ln_1.__class__.__name__}")
        print(f"Layer {i} MLP has norm strategy {trainer.model.transformer.h[i].ln_2.__class__.__name__}")
    trainer.train()

    # Pring final result
    if trainer.is_main_process:
        final_loss = round(trainer.best_val_loss.item(), 4)
        print(f"Best validation loss: {final_loss}")
        # Log the final best validation loss to wandb summary
        wandb.summary["best_val_loss"] = final_loss

    # Finish the wandb run
    wandb.finish()

import os
from typing import Dict

import click
import torch
import torch.nn as nn
import wandb
from diffusers.training_utils import EMAModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from flownav.training.evaluate import evaluate
from flownav.training.train import train


def main_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
) -> None:
    # Set saving paths
    latest_path = os.path.join(project_folder, "latest.pth")

    # Create EMA model
    ema_model = EMAModel(model=model, power=0.75)

    # Run the epochs
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            click.echo(
                click.style(
                    f"> Start epoch {epoch}/{current_epoch + epochs - 1}",
                    fg="magenta",
                )
            )
            train(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        # Save the model, EMA model, optimizer, and scheduler
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, "ema_latest.pth")
        torch.save(ema_model.averaged_model.state_dict(), latest_path)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)

        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, "optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, "scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        # In case of evaluation
        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                click.echo(
                    click.style(
                        f"> Evaluating {dataset_type} dataset at epoch {epoch}",
                        fg="blue",
                    )
                )
                loader = test_dataloaders[dataset_type]
                evaluate(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )

        # Log the current learning rate
        if use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                },
                commit=False,
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

    # Flush the last set of eval logs
    if use_wandb:
        wandb.log({})

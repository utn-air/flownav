import argparse
import os
import time

import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from flownav.data.vint_dataset import ViNT_Dataset
from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from flownav.training.loop import main_loop
from warmup_scheduler import GradualWarmupScheduler


def main(config: dict) -> None:
    # Set up the device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        click.echo(
            click.style(f">> Using GPUs: {config['gpu_ids']}", fg="green", bold=True)
        )
    else:
        click.echo(click.style(">> No GPUs available, using CPU", fg="red", bold=True))
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Set seed for reproducibility
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True

    # Set up the transformation for the dataset (from ImageNet)
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the data
    train_dataset = []
    test_dataloaders = {}
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=True,
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset
    train_dataset = ConcatDataset(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=False,
    )
    click.echo(
        click.style(
            f">> Loaded {len(train_dataset)} training samples",
            fg="cyan",
            bold=True,
        )
    )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset=dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        click.echo(
            click.style(
                f">> Loaded {len(dataset)} test samples for {dataset_type}",
                fg="cyan",
                bold=True,
            )
        )

    # Create the model
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epochs"]
    )
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=1,
        total_epoch=config["warmup_epochs"],
        after_scheduler=scheduler,
    )

    # Load pre-trained model if specified
    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        click.echo(
            click.style(
                f">> Loading pre-trained model from {load_project_folder}",
                fg="yellow",
            )
        )
        if os.path.isdir(load_project_folder):
            latest_path = os.path.join(load_project_folder, "latest.pth")
        elif os.path.isfile(load_project_folder):
            latest_path = load_project_folder
        else:
            click.echo(
                click.style(
                    f">> Could not find pre-trained model at {load_project_folder}",
                    fg="red",
                )
            )
        latest_checkpoint = torch.load(latest_path)
        if "model" in latest_checkpoint:
            model.load_state_dict(latest_checkpoint["model"], strict=True)
        else:
            model.load_state_dict(latest_checkpoint, strict=True)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    # Load Depth-Anything pre-trained weights
    checkpoint = torch.load(
        config["depth"]["weights_path"],
        map_location=device,
    )
    saved_state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    updated_state_dict = {
        k.replace("pretrained.", ""): v
        for k, v in saved_state_dict.items()
        if "pretrained" in k
    }
    new_state_dict = {
        k: v
        for k, v in updated_state_dict.items()
        if k in model.vision_encoder.depth_encoder.state_dict()
    }
    model.vision_encoder.depth_encoder.load_state_dict(new_state_dict, strict=False)

    # Multi-GPU setup
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    # Run the training loop
    main_loop(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )
    click.echo(
        click.style(
            f">> Training completed. Model saved to {config['project_folder']}",
            fg="green",
            bold=True,
        )
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="config/flownav.yaml",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load the configuration
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{this_file_dir}/flownav/config/flownav.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    click.echo(click.style(f">> Using config file: {args.config}", fg="yellow"))

    # Create the project folder and update the configuration
    config.update(user_config)
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config["project_folder"],
    )
    click.echo(
        click.style(
            f">> Project folder created: {config['project_folder']}", fg="yellow"
        )
    )

    # Set wandb configuration
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity=config["entity"],
        )
        wandb.save(args.config, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    main(config)

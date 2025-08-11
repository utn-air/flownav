import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import wandb
import yaml
from diffusers.training_utils import EMAModel
from flownav.visualizing.plot import plot_trajs_and_points


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def load_data_stats() -> dict:
    with open(
        os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    ) as f:
        data_config = yaml.safe_load(f)
    action_stats = {}
    for key in data_config["action_stats"]:
        action_stats[key] = np.array(data_config["action_stats"][key])
    return action_stats


ACTION_STATS = load_data_stats()


def action_reduce(
    unreduced_loss: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, (
        f"{unreduced_loss.shape} != {action_mask.shape}"
    )
    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)


def compute_losses(
    ema_model: EMAModel,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    # Get pred horizon and action dimension
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # Get model output
    model_output_dict = model_output(
        model=ema_model,
        batch_obs_images=batch_obs_images,
        batch_goal_images=batch_goal_images,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_samples=1,
        device=device,
        use_wandb=use_wandb,
    )
    uc_actions = model_output_dict["uc_actions"]
    gc_actions = model_output_dict["gc_actions"]
    gc_distance = model_output_dict["gc_distance"]

    # Compute losses
    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, (
        f"{uc_actions.shape} != {batch_action_label.shape}"
    )
    assert gc_actions.shape == batch_action_label.shape, (
        f"{gc_actions.shape} != {batch_action_label.shape}"
    )

    # Compute action losses
    uc_action_loss = action_reduce(
        F.mse_loss(uc_actions, batch_action_label, reduction="none"), action_mask
    )
    gc_action_loss = action_reduce(
        F.mse_loss(gc_actions, batch_action_label, reduction="none"), action_mask
    )

    uc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1),
        action_mask,
    )
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )
    gc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1),
        action_mask,
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats) -> np.ndarray:
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_delta(actions: np.ndarray) -> np.ndarray:
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action(ndeltas, action_stats=ACTION_STATS) -> torch.Tensor:
    device = ndeltas.device
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    # Exploration
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    # Navigation
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    with torch.no_grad():
        start_time = time.time()

        # Exploration
        output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
            ),
            output,
            torch.linspace(0, 1, 10, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        uc_actions = get_action(traj[-1], ACTION_STATS)
        proc_time = time.time() - start_time
        mean_proc_time = proc_time / output.shape[0]
        if use_wandb:
            wandb.log({"Mean Processing Time UC": mean_proc_time})
            wandb.log({"Processing Time UC": proc_time})

        # Navigation
        output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obsgoal_cond
            ),
            output,
            torch.linspace(0, 1, 10, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        gc_actions = get_action(traj[-1], ACTION_STATS)
        proc_time = time.time() - start_time
        mean_proc_time = proc_time / output.shape[0]
        if use_wandb:
            wandb.log({"Mean Processing Time GC": mean_proc_time})
            wandb.log({"Processing Time GC": proc_time})

    # Predict distance
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def visualize_action_distribution(
    ema_model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
) -> None:
    # Create visualization directory
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    # Select visualization samples
    max_batch_size = batch_obs_images.shape[0]
    num_images_log = min(
        num_images_log,
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_action_label.shape[0],
        batch_goal_pos.shape[0],
    )
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    # Prepare Weights and Biases logging
    wandb_list = []
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # Split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)
    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    # Run
    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            model=ema_model,
            batch_obs_images=obs,
            batch_goal_images=goal,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            num_samples=num_samples,
            device=device,
            use_wandb=use_wandb,
        )
        uc_actions_list.append(to_numpy(model_output_dict["uc_actions"]))
        gc_actions_list.append(to_numpy(model_output_dict["gc_actions"]))
        gc_distances_list.append(to_numpy(model_output_dict["gc_distance"]))

    # Concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # Split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)
    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]
    np_distance_labels = to_numpy(batch_distance_labels)

    # Plot
    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])
        traj_list = np.concatenate(
            [
                uc_actions,
                gc_actions,
                action_label[None],
            ],
            axis=0,
        )
        traj_colors = (
            ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        )
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]
        plot_trajs_and_points(
            ax=ax[0],
            list_trajs=traj_list,
            list_points=point_list,
            traj_colors=traj_colors,
            point_colors=point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
        )
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)
        ax[0].set_title("action predictions")
        ax[1].set_title("observation")
        ax[2].set_title(
            f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}"
        )

        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)

    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)

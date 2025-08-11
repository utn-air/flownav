import itertools

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import tqdm
import wandb
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchvision import transforms
from flownav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from flownav.training.logger import Logger
from flownav.training.utils import (
    ACTION_STATS,
    compute_losses,
    get_delta,
    normalize_data,
    visualize_action_distribution,
    from_numpy,
)


def evaluate(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    # Clip goal mask probability
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()

    # Get number of batches
    num_batches = len(dataloader)

    # Set up loggers
    uc_action_loss_logger = Logger(
        "uc_action_loss", eval_type, window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", eval_type, window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        dynamic_ncols=True,
        desc=f"Evaluating {eval_type} for epoch {epoch}",
        leave=True,
        colour="blue",
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                _,
                action_mask,
            ) = data

            # Split the observation image into RGB channels
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(
                obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_viz_goal_images = TF.resize(
                goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            # Get batch size
            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)

            # Create exploration mask
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)

            # Create navigation mask
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            # Get random mask conditioning
            rand_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=rand_goal_mask,
            )

            # Get navigation conditioning
            obsgoal_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=no_mask,
            )
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            # Get exploration conditioning
            goal_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # get distance to goal
            distance = distance.to(device)

            # Get naction and normalize it
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Flow
            FM = ConditionalFlowMatcher(sigma=0.0)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)

            # 1. RANDOM MASK
            rand_mask_vt = ema_model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=rand_mask_cond
            )

            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_vt, ut)

            # 2. No mask
            no_mask_vt = ema_model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond
            )

            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_vt, ut)

            # 3. Goal mask
            goal_mask_vt = ema_model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=goal_mask_cond
            )

            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_vt, ut)

            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            if use_wandb:
                wandb.log({"flownav_eval_loss (random masking)": rand_mask_loss})
                wandb.log({"flownav_eval_loss (no masking)": no_mask_loss})
                wandb.log({"flownav_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = compute_losses(
                    ema_model=ema_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_dist_label=distance.to(device),
                    batch_action_label=actions.to(device),
                    device=device,
                    action_mask=action_mask.to(device),
                    use_wandb=use_wandb,
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_action_distribution(
                    ema_model=ema_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_viz_obs_images=batch_viz_obs_images,
                    batch_viz_goal_images=batch_viz_goal_images,
                    batch_action_label=actions,
                    batch_distance_labels=distance,
                    batch_goal_pos=goal_pos,
                    device=device,
                    eval_type="train",
                    project_folder=project_folder,
                    epoch=epoch,
                    num_images_log=num_images_log,
                    num_samples=4,
                    use_wandb=use_wandb,
                )

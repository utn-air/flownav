
import os
import numpy as np
import torch
import yaml
from cv_bridge import CvBridge
import cv2
import pickle
from PIL import Image as PILImage
import argparse
import torchdiffeq
from pathlib import Path

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy

# ROS Topics
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC,
                        REACHED_GOAL_TOPIC)

# Custom Imports
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_model


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 


# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class NavigationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('Navigation_Node')

        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        self.context_size = None
        self.context_queue = []

        self.cur_img = None
        self.cur_naction = None

        self.k_steps = args.k_steps

        ckpt_path = Path(args.ckpt)
        self.cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.goal_node}_{args.k_steps}"
        os.makedirs(self.cur_exp_dir, exist_ok=True)

        self.cur_exp_im_dir = f"{self.cur_exp_dir}/images"
        os.makedirs(self.cur_exp_im_dir, exist_ok=True)

        self.cur_exp_pkl_dir = f"{self.cur_exp_dir}/pkl"
        os.makedirs(self.cur_exp_pkl_dir, exist_ok=True)

        self.im_idx = 0

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

        self.context_size = model_params["context_size"]

        # load model weights
        ckpth_path = args.ckpt
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            model_params,
            device,
        )
        self.model = self.model.to(device)
        self.model.eval()

        # load topomap
        topomap_filenames = sorted(os.listdir(os.path.join(
            TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
        num_nodes = len(os.listdir(topomap_dir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        closest_node = 0
        assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
        if args.goal_node == -1:
            goal_node = len(topomap) - 1
        else:
            goal_node = args.goal_node
        self.reached_goal = False

        # ROS 2
        self.image_sub = self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self.callback_obs, qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                            history=QoSHistoryPolicy.KEEP_LAST,
                                                                            depth=10))
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                            history=QoSHistoryPolicy.KEEP_LAST,
                                                                            depth=10))
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                            history=QoSHistoryPolicy.KEEP_LAST,
                                                                            depth=10))
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.timer = self.create_timer(1.0 / RATE, lambda: self.run_navigation_loop(args))

        self.imsave_timer = self.create_timer(1, lambda:self.save_images_and_actions())

        print("Waiting for image observations...")

        self.model_params = model_params

        self.closest_node = closest_node
        self.goal_node = goal_node
        self.topomap = topomap
        self.br = CvBridge()

    def callback_obs(self, msg: Image):
        self.get_logger().info("Reached Image callback!")
        self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        self.obs_img = PILImage.fromarray(cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB))

        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(self.obs_img)

    def save_images_and_actions(self):
        if self.cur_img is not None and self.cur_naction is not None:
            print(f"Saving Image and action {self.im_idx}")
            self.cur_img.save(f"{self.cur_exp_im_dir}/{self.im_idx}.png")
            
            with open(f"{self.cur_exp_pkl_dir}/{self.im_idx}.pkl", "wb") as f:
                pickle.dump(self.cur_naction, f)
                
            self.im_idx += 1

    def run_navigation_loop(self, args):
        chosen_waypoint = np.zeros(4)

        if len(self.context_queue) > self.context_size:

            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device)  

            start = max(self.closest_node - args.radius, 0)
            end = min(self.closest_node + args.radius + 1, self.goal_node)
            goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device) for g_img in self.topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            self.closest_node = min_idx + start

            # infer action
            with torch.no_grad():
                obs_cond = obsgoal_cond[min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)

                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                noisy_action = torch.randn(
                    (args.num_samples, self.model_params["len_traj_pred"], 2), device=device)

                traj = torchdiffeq.odeint(
                    lambda t, x: self.model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
                    noisy_action,
                    torch.linspace(0, 1, self.k_steps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                )
                naction = traj[-1]
                    
                naction = to_numpy(get_action(naction))

                # Save for logging
                self.cur_naction = naction
                self.cur_img = self.context_queue[-1]

                sampled_actions_msg = Float32MultiArray()
                message_data = np.concatenate((np.array([0]), naction.flatten()))
                sampled_actions_msg.data = message_data.tolist()
                print("published sampled actions")
                self.sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
            
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.waypoint_pub.publish(waypoint_msg)

        print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

        reached_goal = self.closest_node == self.goal_node
        goal_reached_msg = Bool()
        goal_reached_msg.data = bool(reached_goal)
        self.goal_pub.publish(goal_reached_msg)
        
        if reached_goal:
            print("Reached goal! Stopping...")
        

def main(args: argparse.Namespace):
    rclpy.init()
    navigation_node = NavigationNode(args)

    try:
        rclpy.spin(navigation_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run FlowNav Navigation on the turtlebot")
    
    parser.add_argument(
        "--model",
        "-m",
        default="flownav",
        type=str,
        help="Model to run: Only FlowNav is supported currently (default: flownav)",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of the waypoint used for navigation (default: 2)",
    )
    parser.add_argument(
        "--k_steps",
        "-k",
        default=10, 
        type=int,
        help="Number of time steps",
    )
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (default: -1)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="temporal distance within the next node in the topomap before localizing to it",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="temporal number of locobal nodes to look at in the topopmap for localization",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model",
    )
    parser.add_argument(
        "--exp_dir",
        "-d",
        default="./nav_experiments",
        type=str,
        help="Path to log experiment results",
    )

    args = parser.parse_args()

    print(f"Using {device}")
    main(args)
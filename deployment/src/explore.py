import os
import shutil
import numpy as np
import cv2
import yaml
import torchdiffeq
import torch
from PIL import Image as PILImage
import argparse
import time
import pickle
from pathlib import Path

from cv_bridge import CvBridge

# ROS2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  CompressedImage
from std_msgs.msg import  Float32MultiArray
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy

# Ros Topics
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

# Custom Imports
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_model, remove_files_in_dir

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
TOPOMAP_IMAGES_DIR = "../topomaps/images"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class Exploration(Node):
    def __init__(self, dir_name: str, dt: float, args: argparse.Namespace):
        super().__init__('exploration_node')

        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        self.context_size = None
        self.context_queue = []

        self.cur_img = None
        self.cur_naction = None

        ckpt_path = Path(args.ckpt)
        self.cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.k_steps}"
        os.makedirs(self.cur_exp_dir, exist_ok=True)

        self.cur_exp_im_dir = f"{self.cur_exp_dir}/images"
        os.makedirs(self.cur_exp_im_dir, exist_ok=True)

        self.cur_exp_pkl_dir = f"{self.cur_exp_dir}/pkl"
        os.makedirs(self.cur_exp_pkl_dir, exist_ok=True)

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        # load model weights
        ckpth_path = args.ckpt 
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            self.model_params,
            device,
        )
        self.model = self.model.to(device)
        self.model.eval()

        closest_node = 0
        self.closest_node = closest_node
        self.im_idx = 0

        # ROS2 nodes

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
        
        self.timer = self.create_timer(1.0 / RATE, lambda: self.run_exploration_loop(args))
        self.imsave_timer = self.create_timer(1, lambda:self.save_images_and_actions())

        # topomap timer
        self.obs_img = None
        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, dir_name)
        self.dt = dt
        self.img_idx = 0
        self.br = CvBridge()

        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            print(f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)
            
        self.get_logger().info("Waiting for images...")


    def timer_callback_topomap(self):
        if self.obs_img is not None:
            self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.img_idx}.png"))
            self.get_logger().info(f"Published image {self.img_idx}")
            self.img_idx += 1
            self.obs_img = None

    def save_images_and_actions(self):
        if self.cur_img is not None and self.cur_naction is not None:
            print(f"Saving Image and action {self.im_idx}")
            self.cur_img.save(f"{self.cur_exp_im_dir}/{self.im_idx}.png")
            
            with open(f"{self.cur_exp_pkl_dir}/{self.im_idx}.pkl", "wb") as f:
                pickle.dump(self.cur_naction, f)
                
            self.im_idx += 1

    def run_exploration_loop(self, args):
        start_time = time.time()

        waypoint_msg = Float32MultiArray()
        if len(self.context_queue) > self.model_params["context_size"]:

            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device) # ignore the goal

            with torch.no_grad():

                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
                # encoder vision features
                obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                
                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                traj = torchdiffeq.odeint(
                    lambda t, x: self.model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
                    noisy_action,
                    torch.linspace(0, 1, args.k_steps, device=device),
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
                sampled_action_message_data = np.concatenate((np.array([0]), naction.flatten()))
                sampled_actions_msg.data = sampled_action_message_data.tolist()
                self.sampled_actions_pub.publish(sampled_actions_msg)

                naction = naction[0] # change this based on heuristic

                chosen_waypoint = naction[args.waypoint]

                if self.model_params["normalize"]:
                    chosen_waypoint *= (MAX_V / RATE)
                waypoint_msg.data = chosen_waypoint.flatten().tolist()
                self.waypoint_pub.publish(waypoint_msg)
                print("Published waypoint")

        print("time elapsed:", time.time() - start_time)
        

    def callback_obs(self, msg):
        self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)

        self.obs_img = PILImage.fromarray(cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB))

        self.current_image = np.array(self.obs_img)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(self.obs_img)


def main(args: argparse.Namespace):
    rclpy.init()
    exploration_node = Exploration(dir_name=args.dir, dt=args.dt, args=args)

    try:
        rclpy.spin(exploration_node)
    except KeyboardInterrupt:
        pass
    finally:
        exploration_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run FlowNav on Exploration on a turtebot")
    parser.add_argument(
        "--model",
        "-m",
        default="flownav",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--k_steps",
        "-k",
        default=10,
        type=int,
        help="Number of time steps",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--exp_dir",
        "-s",
        default="explore_topomap",
        type=str,
        help="Path to store the exploration topomap",
    )
    parser.add_argument(
        "--dir",
        default="test",
        type=str,
        help="Name of the specific exploration topomap or the experiment",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="checkpoint_path",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)



import argparse
import os
import shutil
import time
from utils import msg_to_pil
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from topic_names import IMAGE_TOPIC

TOPOMAP_IMAGES_DIR = "../topomaps/images"
IMAGE_TOPIC = '/image_compressed'

def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class TopoMapNode(Node):
    def __init__(self, dir_name: str, dt: float):
        super().__init__("create_topomap")
        self.subscriber_image = self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self.callback_obs, 10)
        self.obs_img = None
        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, dir_name)
        self.dt = dt
        self.i = 0
        self.start_time = time.time()
        self.br = CvBridge()

        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            print(f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)

        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info("Waiting for images...")


    def callback_obs(self, msg: Image):
        # self.obs_img = msg_to_pil(msg)
        self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        self.obs_img = PILImage.fromarray(cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB))

    def timer_callback(self):
        if self.obs_img is not None:
            self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
            self.get_logger().info(f"Published image {self.i}")
            self.i += 1
            self.start_time = time.time()
            self.obs_img = None

        if time.time() - self.start_time > 10 * self.dt:
            self.get_logger().warning(f"Topic {IMAGE_TOPIC} not publishing anymore")

            pass


def main(args: argparse.Namespace):
    rclpy.init()
    node = TopoMapNode(args.dir, args.dt)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="Path to store topomap in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)",
    )
    args = parser.parse_args()

    main(args)

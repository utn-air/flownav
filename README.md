# FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation

[![Project Page](https://img.shields.io/badge/Project%20Page-6cc644&cacheSeconds=60)](https://utn-air.github.io/flownav.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09524-b31b1b.svg)](https://arxiv.org/abs/2411.09524)
![GitHub License](https://img.shields.io/github/license/utn-air/flownav?label=License&color=%23e11d48&cacheSeconds=60)
[![Latest Release](https://img.shields.io/github/v/tag/utn-air/flownav?label=Latest%20Release&color=%4078c0&cacheSeconds=60)
](https://github.com/utn-air/flownav/releases)

> [!NOTE]  
> This source code was based on [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html).

## 🪛 Installation

1) Create a virtual environment with python 3.10 or higher. 
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the dependencies using pip or pdm

```bash
pip install -r requirements.txt
# or (choose one)
pdm install
```

## 🔗 Download data and weights

1) Download and process the datasets according to [NoMaD's instructions](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling)
2) Download [DepthAnything-V2 ViT-s weights](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)
3) If you want to use the pretrained model, download the weights from [our latest release](https://github.com/utn-air/flownav/releases). 

## 🚀 Training

To train the model, you need to adjust the in the configuration YAML [`flownav.yaml`](flownav/config/flownav.yaml) as:

1) `train` to `True`
2) `depth/weights_path` to the DepthAnythingV2 checkpoint path
3) `datasets/<DATASET>/data_folder`, `datasets/<DATASET>/train` and datasets/<DATASET>/`test` to the folders generated during the (data processing step)[#-download-data-and-weights].

Then, run the following command:

```bash
python train.py -c <YOUR_CONFIG>.yaml
```

If you want to use [wandb](https://wandb.ai/) to log the training, you can set the `wandb_logging` flag in the configuration YAML to `True` and  the `project` and `entity` to your desired project and entity (usually your username). Don't forget to login first:
    
```bash
wandb login
```

## 📊 Testing

To test the model, you need to have the model trained. Weights are available in the [latest release](https://github.com/utn-air/flownav/releases). Adjust in the configuration YAML [`flownav.yaml`](flownav/config/flownav.yaml) as:

1) `train` to `False`.
2) `depth/weights_path` to the DepthAnythingV2 checkpoint path
2) `load_run` to the path of the desired weights.

Then, run the following command:

```bash
python train.py -c <YOUR_CONFIG>.yaml
```

## 🤖 Deployment

> [!WARNING]
> <span style="color:red">The deployment code is still being refactored. There may be issues with the current code.</span>

> [!NOTE]
> The deployment code is based on [NoMaD's deployment code](https://github.com/robodhruv/visualnav-transformer/tree/main?tab=readme-ov-file#deployment). However, we move to ROS2 and use the [TurtleBot 4](https://turtlebot.github.io/turtlebot4-user-manual/) as the robot platform.

The `deployment` subfolder contains the code to deploy FlowNav on a [TurtleBot 4](https://turtlebot.github.io/turtlebot4-user-manual/).

### TurtleBot4 and Edge Device Setup
FlowNav was tested on Ubuntu 22.04. Both the turtlebot and edge device (laptop with an NVIDIA A500 GPU) run ROS 2 Humble. Communication between the two devices happens over a Wi-Fi network. The turtlebot runs the Discovery Server mode, which uses an internal USB-C connection between the Raspberry Pi and the low-level hardware to ensure low latency between receiving an action and executing the action.

#### Software
- ROS 2 Humble: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).
- TurtleBot 4 packages: 
  - Install the TurtleBot 4 packages by following the [official installation guide](https://turtlebot.github.io/turtlebot4-user-manual/setup/basic.html)
  - Discovery Server mode of the turtlebot: Follow this [setup guide](https://turtlebot.github.io/turtlebot4-user-manual/setup/discovery_server.html) to enable the Discovery Server mode
  - Make sure you have already created and activated the virtual environment and installed the required packages as described in the [Installation](#-installation) section.
  - Install the required packages:
    ```bash
    pip install -e flownav/
    ```
  - Install the [Conditional Flow Matching](https://github.com/atong01/conditional-flow-matching#) library:
    ```bash
    git clone git@github.com:atong01/conditional-flow-matching.git
    pip install -e conditional-flow-matching/
    ```
- Install [tmux](https://github.com/tmux/tmux/wiki/Installing) to run multiple terminal sessions in the same window

#### Hardware

- TurtleBot 4: Follow the [official user manual](https://turtlebot.github.io/turtlebot4-user-manual/) to set up the TurtleBot 4.
- Wide angle RGB Camera: 
- Joystick or keyboard for teleoperation: 
  - If you want to use a joystick, follow the [official guide](https://turtlebot.github.io/turtlebot4-user-manual/setup/basic.html#turtlebot-4-controller-manual-setup) to set it up.

### Update config files
- Update the topic names in the `deployment/src/topic_names.py` file to match your setup.
- Update the model config paths in `deployment/config/models.yaml`. 

### Running FlowNav

#### Navigation Mode
- Create a topological map using the following script. The topological map is saved in `topomaps/<topomap_directory>`.
```bash
cd deployment/src/navigation
./create_topomap.sh <topomap_directory>
```
The command opens up two tmux windows: One to collect images for the topological map and another to control the robot using teleoperation. You can use a joystick or keyboard for teleoperation.
- Run the Navigation Node:
```bash
cd deployment/src/navigation
./navigate.sh "--model <model_name> --ckpt <ckpt_path> --dir <topomap_directory>"
```
Only `flownav` is supported for the model. Adjust other argparse arguments as needed.

#### Exploration Mode
Exploration does not require a topological map.
- Run the Exploration Node:
```bash
cd deployment/src/exploration
./explore.sh "--model <model_name> --ckpt <ckpt_path> --exp_dir <save_directory>"
```
Only `flownav` is supported for the model. Adjust other argparse arguments as needed.

## 📝 Citation

```
@misc{gode2025flownav,
      title={FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation}, 
      author={Gode, Samiran and Nayak, Abhijeet and Oliveira, Débora N.P. and Krawez, Michael 
              and Schmid, Cordelia and Burgard, Wolfram},
      year={2025},
      eprint={2411.09524},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.09524}, 
}
```

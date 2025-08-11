# FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation

[![Project Page](https://img.shields.io/badge/Project%20Page-6cc644&cacheSeconds=60)](https://utn-air.github.io/flownav.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09524-b31b1b.svg)](https://arxiv.org/abs/2411.09524)
[![GitHub License](https://img.shields.io/github/license/utn-air/flownav?label=License&color=%23e11d48&cacheSeconds=3600)
](https://github.com/utn-air/flownav/blob/main/LICENSE)
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

We deployed to a TurtleBot 4 using ROS2 Humble. 

> [!WARNING]  
> **Code will be uploaded soon.**

## 📝 Citation

```
@misc{gode2025flownav,
      title={FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation}, 
      author={Gode, Samiran and Nayak, Abhijeet and Oliveira, Débora N.P. and  and Krawez, Michael 
              and Schmid, Cordelia and Burgard, Wolfram},
      year={2025},
      eprint={2411.09524},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.09524}, 
}
```

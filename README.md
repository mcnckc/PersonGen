# Aligning Diffusion Model by HumanPreference

This repository contains the official implementation of the **ImageReFL** method:

## Overview

This codebase implements **ImageReFL**, two approaches that significantly improve diversity and visual quality in reward fine-tuning of diffusion models.

We provide training and inference scripts for:
- Standard ReFL
- ReFL with Combined Generation
- ImageReFL
- Evaluation metrics and user study framework

Supported base models:
- Stable Diffusion 1.5 (SD1.5)
- Stable Diffusion XL (SDXL)

Reward models:
- HPSv2.1
- PickScore

## Installation

```bash
git clone https://github.com/TorchRik/HumanDiffusion.git
cd HumanDiffusion
#<-> create your favourite venv
pip install requirments.txt
```

## Training

To run the ImageReFL fine-tuning algorithm with default parameters:
```bash
HYDRA_FULL_ERROR=1 python train.py --config-name image_refl_train
```
To run the standard ReFL algorithm:

```bash
HYDRA_FULL_ERROR=1 python train.py --config-name refl_train
```

You can also customize parameters using Hydra syntax. For example:
```bash
HYDRA_FULL_ERROR=1 python train.py writer.run_name='wandb_run_name' --config-name image_refl_train
```

## Inference
To run inference with a trained model and compute image quality metrics, use:

```bash
HYDRA_FULL_ERROR=1 python inference.py inferencer.from_pretrained="saved/<your_train_run_name>/checkpoint-epoch20.pth"
```

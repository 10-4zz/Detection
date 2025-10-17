# A project for detection models in computer vision by PyTorch.

## content
- [Introduction](#introduction)
- [Setup](#setup)

## Introduction
This project uses PyTorch to implement current mainstream and classic computer vision detection models, including but not limited to the following:
- Yolo series

In addition, some small tools for dataset conversion are implemented.

## Setup

Setup a conda environment:
````
conda create -n detection python==3.12 -y
conda activate detection
````

Install PyTorch and torchvision(cuda>=11.8):
````
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
````

Install related dependencies:
````
pip install -r requirements.txt
````

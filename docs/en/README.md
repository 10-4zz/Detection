# A project for detection models in computer vision by PyTorch.

## Content
- [Introduction](#introduction)
- [Setup](#setup)

## Introduction
This document will introduce how to train model from zero through the project.

## Setup
Clone this project from github to your own workspace:
````
# git clone https://github.com/10-4zz/Detection.git
git clone git@github.com:10-4zz/Detection.git
cd Detection
````
Note: please select the https or ssh according to your situation.

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

After completing the above steps, you can run the command as below to check the dependencies and cuda version for environment in terminal.
````
bash scripts/check_env.sh
````
If all dependencies are correctly installed, you will see the message "All specified package versions are installed and correct.".

If the CUDA version in the environment meets the requirements, you will see the message "Your NVIDIA driver version is sufficient for the CUDA runtime version used by PyTorch."

If you want to use other version of torch, you can refer to the official website: https://pytorch.org/get-started/previous-versions/.

And you also can use other cuda version, but it may occur conflict in different packages(Different torch version also can make your environment get conflict).


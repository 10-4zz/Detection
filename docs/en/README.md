# A project for detection models in computer vision by PyTorch.

## Content
- [Introduction](#introduction)
- [Setup](#setup)

## Introduction
This project uses PyTorch to implement current mainstream and classic computer vision detection models, including but not limited to the following:
- Yolo series

In addition, some small tools for dataset conversion are implemented.

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
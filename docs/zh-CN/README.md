# 用Pytorch实现的计算机视觉检测模型

## 目录
- [简介](#简介)
- [安装](#安装)

## 简介
本项目使用Pytorch实现了目前主流或者经典的计算机视觉检测模型，包含但不限于以下模型：
- Yolo系列

此外，实现了一些数据集转换的小工具。


## 安装
从github克隆本项目到工作目录下:
````
# git clone https://github.com/10-4zz/Detection.git
git clone git@github.com:10-4zz/Detection.git
cd Detection
````
注: 根据情况来选择https或者ssh

创建一个新的conda环境：
````
conda create -n detection python==3.12 -y
conda activate detection
````

安装PyTorch和torchvision(cuda>=11.8):
````
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
````

安装相关依赖:
````
pip install -r requirements.txt
````

在完成上述步骤之后，你可以在终端运行下面的命令来检查环境中的依赖和cuda版本：
````
bash scripts/check_env.sh
````
如果所有的依赖都安装完毕且版本正确，将会看到“All specified package versions are installed and correct.”的信息。

如果环境中的cuda版本满足需求，将会看到“Your NVIDIA driver version is sufficient for the CUDA runtime version used by PyTorch.”的信息。


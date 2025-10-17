# 用Pytorch实现的计算机视觉检测模型

## 目录
- [简介](#简介)
- [安装](#安装)

## 简介
本项目使用Pytorch实现了目前主流或者经典的计算机视觉检测模型，包含但不限于以下模型：
- Yolo系列

此外，实现了一些数据集转换的小工具。


## 安装
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


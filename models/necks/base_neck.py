"""
Writen by: ian
"""
import torch.nn as nn


class BaseNeck(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Please Implement forward method")

    def get_feat_index(self):
        raise NotImplementedError("Please Implement get_feat_index method")

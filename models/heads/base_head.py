"""
Writen by: ian
"""
import torch.nn as nn


class BaseHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


"""
Writen by: ian
"""
from typing import List

import torch
import torch.nn as nn

from models.necks.base_neck import BaseNeck
from models.components.yolo import CBS, C3

from utils.logger import logger


class FPN(BaseNeck):
    """
    This FPN is used for yolo.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            layer_num: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.layer_num = layer_num

        self.fpn_layers = self._make_fpn_layers()

    def forward(self, x):
        if len(x) != self.layer_num:
            logger.error("The length of input features must be equal to layer_num")


    def _make_fpn_layers(self) -> List:
        layers = []
        for _ in range(self.layer_num):
            conv_layer = CBS(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )

            up_sample_layer = nn.Upsample(
                size=None,
                scale_factor=2,
                mode='nearest'
            )

            concat_layer = torch.Concat

            block = C3(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                num_blocks=3,
                shortcut=False,
            )
            layer = {
                "conv_layer": conv_layer,
                "up_sample_layer": up_sample_layer,
                "concat_layer": concat_layer,
                "block": block,
            }
            layers.append(layer)

        return layers




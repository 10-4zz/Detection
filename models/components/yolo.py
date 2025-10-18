"""
Writen by: ian
"""
import warnings

import torch
import torch.nn as nn
from torch import Tensor

from models.utils import auto_pad


class CBS(nn.Module):
    default_act = nn.SiLU()
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = None,
            groups: int = 1,
            dilation: int = 1,
            use_act: bool = True,
            act_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size=kernel_size, padding=padding, dilation=dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        if act_layer is not None:
            self.default_act = act_layer
        self.act_layer = self.default_act if use_act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act_layer(self.batch_norm(self.conv_layer(x)))

    def inference(self, x: Tensor) -> Tensor:
        return self.act_layer(self.conv_layer(x))


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            shortcut: bool = True,
            groups: int = 1,
            expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv_layer1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.conv_layer2 = CBS(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            groups=groups,
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_layer2(self.conv_layer1(x)) if self.shortcut else self.conv_layer2(self.conv_layer1(x))

    def inference(self, x: Tensor) -> Tensor:
        if self.shortcut:
            x1 = self.conv_layer1.inference(x)
            x2 = self.conv_layer2.inference(x1)
            out = x2 + x
        else:
            x1 = self.conv_layer1.inference(x)
            x2 = self.conv_layer2.inference(x1)
            out = x2
        return out


class C3(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv_layer1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.conv_layer2 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.conv_layer3 = CBS(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.module = nn.Sequential(
            *(
                Bottleneck(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layer3(torch.cat((self.module(self.conv_layer1(x)), self.conv_layer2(x)), dim=1))

    def inference(self, x: Tensor) -> Tensor:
        x1 = self.conv_layer1.inference(x)
        x2 = self.conv_layer2.inference(x)
        for block in self.module:
            x1 = block.inference(x1)
        out = self.conv_layer3.inference(torch.cat((x1, x2), dim=1))
        return out


class SpatialPyramidPoolingFast(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv_layer1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.conv_layer2 = CBS(
            in_channels= 4 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layer1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.max_pool(x)
            y2 = self.max_pool(y1)
            return self.conv_layer2(torch.cat((x, y1, y2, self.max_pool(y2)), dim=1))

    def inference(self, x: Tensor) -> Tensor:
        x = self.conv_layer1.inference(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.max_pool(x)
            y2 = self.max_pool(y1)
            return self.conv_layer2.inference(torch.cat((x, y1, y2, self.max_pool(y2)), dim=1))
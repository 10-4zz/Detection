"""
Writen by: ian
"""
import torch.nn as nn
from torch import Tensor

from models.backbones.base_backbone import BaseBackbone
from models.backbones import BACKBONES_REGISTRY

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


@BACKBONES_REGISTRY.register(component_name="csp_darknet53")
class CSPDarkNet53(BaseBackbone):
    def __init__(self) -> None:
        super().__init__()


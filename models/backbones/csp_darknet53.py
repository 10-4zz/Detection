"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Dict, Any, Optional, Union, List, Tuple

import torch.nn as nn
from torch import Tensor

from models.backbones.base_backbone import BaseBackbone
from models.backbones import BACKBONES_REGISTRY
from models.components.yolo import CBS, C3, SpatialPyramidPoolingFast

from utils.math_utils import make_divisible, bound_fn


def get_config(width_multiple: float, depth_multiple: float) -> Dict[str, Any]:
    """
    Generates the configuration for the CSPDarkNet53 backbone, optimized for readability and maintainability.
    """
    layer_params = [
        (64, 128, 3),  # for layer1
        (128, 256, 6),  # for layer2
        (256, 512, 9),  # for layer3
        (512, 1024, 3),  # for layer4
    ]

    config = {
        "in_proj": {
            "input_dim": 3,
            "out_channels": int(make_divisible(64 * width_multiple, divisor=16)),
            "kernel_size": 6,
            "stride": 2,
            "block_type": CBS,
        },
    }

    for i, (in_ch_base, out_ch_base, depth_factor) in enumerate(layer_params):
        layer_name = f"layer{i + 1}"
        config[layer_name] = {
            "in_channels": int(make_divisible(in_ch_base * width_multiple, divisor=16)),
            "out_channels": int(make_divisible(out_ch_base * width_multiple, divisor=16)),
            "kernel_size": [3, None],
            "stride": [2, None],
            "num_blocks": [
                1,
                int(bound_fn(min_val=1, max_val=100, value=depth_factor * depth_multiple))
            ],
            "block_type": [CBS, C3],
        }

    config["layer5"] = {
        "in_channels": int(make_divisible(1024 * width_multiple, divisor=16)),
        "out_channels": int(make_divisible(1024 * width_multiple, divisor=16)),
        "kernel_size": [5],
        "num_blocks": [1],
        "block_type": [SpatialPyramidPoolingFast],
    }

    return config

@BACKBONES_REGISTRY.register(component_name="csp_darknet53")
class CSPDarkNet53(BaseBackbone):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)

        width_multiple = getattr(opts, "model.backbone.csp_darknet53.width_multiple", 1.0)
        depth_multiple = getattr(opts, "model.backbone.csp_darknet53.depth_multiple", 1.0)
        cfg = get_config(width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.strides = {
            str(int(64 * width_multiple)): 2,
            str(int(128 * width_multiple)): 4,
            str(int(256 * width_multiple)): 8,
            str(int(512 * width_multiple)): 16,
            str(int(1024 * width_multiple)): 32,
        }

        self.in_proj = cfg["in_proj"]["block_type"](
            in_channels=cfg["in_proj"]["input_dim"],
            out_channels=cfg["in_proj"]["out_channels"],
            kernel_size=cfg["in_proj"]["kernel_size"],
            stride=cfg["in_proj"]["stride"]
        )

        self.layer1 = self.build_layer(cfg["layer1"])
        self.layer2 = self.build_layer(cfg["layer2"])
        self.layer3 = self.build_layer(cfg["layer3"])
        self.layer4 = self.build_layer(cfg["layer4"])
        self.layer5 = self.build_layer(cfg["layer5"])

    @staticmethod
    def build_layer(cfg: Dict) -> nn.Sequential:
        layers = []
        current_in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]

        block_types = cfg["block_type"]
        num_blocks_list = cfg["num_blocks"]
        num_block_types = len(block_types)
        kernel_sizes = cfg.get("kernel_size", [None] * num_block_types)
        strides = cfg.get("stride", [1] * num_block_types)

        for i, (block_cls, num, ks, s) in enumerate(zip(
                block_types, num_blocks_list, kernel_sizes, strides
        )):
            for n in range(num):
                in_ch = current_in_channels if (i == 0 and n == 0) else out_channels

                block_name = block_cls.__name__

                if block_name == 'C3':
                    layer = block_cls(
                        in_channels=in_ch,
                        out_channels=out_channels
                    )
                elif block_name == 'SpatialPyramidPoolingFast':
                    layer = block_cls(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        kernel_size=ks
                    )
                else:
                    stride = s if n == 0 else 1
                    layer = block_cls(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        kernel_size=ks,
                        stride=stride
                    )

                layers.append(layer)

        return nn.Sequential(*layers)

    def inference(self, x: Tensor) -> Tensor:
        if self.in_proj is not None:
            x = self.in_proj.inference(x)
            self.layer_out.append(x)

        if self.layer1 is not None:
            for block in self.layer1:
                x = block.inference(x)
            self.layer_out.append(x)

        if self.layer2 is not None:
            for block in self.layer2:
                x = block.inference(x)
            self.layer_out.append(x)

        if self.layer3 is not None:
            for block in self.layer3:
                x = block.inference(x)
            self.layer_out.append(x)

        if self.layer4 is not None:
            for block in self.layer4:
                x = block.inference(x)
            self.layer_out.append(x)

        if self.layer5 is not None:
            for block in self.layer5:
                x = block.inference(x)
            self.layer_out.append(x)

        return x

    def get_map_size(
            self,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = 640,
    ) -> Dict[str, int]:
        keys = self.strides.keys()
        layer_out_size = {}
        for key in keys:
            layer_out_size[key] = int(input_size / self.strides[key])
        return layer_out_size

    def get_strides(self) -> Dict[str, int]:
        return self.strides

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.backbone.csp_darknet53.width_multiple",
            type=float,
            default=1.0,
            help="The width multiple of the CSPDarkNet53."
        )
        group.add_argument(
            "--model.backbone.csp_darknet53.depth_multiple",
            type=float,
            default=1.0,
            help="The depth multiple of the CSPDarkNet53."
        )

        return parser


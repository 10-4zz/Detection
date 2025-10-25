"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Tuple, List, Any, Dict

import torch
import torch.nn as nn

from models.heads.base_head import BaseHead
from models.heads import HEADS_REGISTRY


@HEADS_REGISTRY.register(component_name='yolov5_head')
class YOLOv5Head(BaseHead):
    """
    The head of YOLOv5.
    """
    stride = None
    dynamic = False
    def __init__(
            self,
            opts: argparse.Namespace,
    ):
        super().__init__(opts)
        anchors = getattr(opts, "model.head.yolov5.anchors", 3)
        self.in_channels_list = getattr(opts, "model.head.yolov5.in_channels", None)

        self.num_outputs = self.num_classes + 5
        self.num_detection_layer = len(anchors)
        self.num_anchors = len(anchors[0]) // 2

        self.detector_head = nn.ModuleList(
            nn.Conv2d(
                in_channels=x,
                out_channels=self.num_outputs * self.num_anchors,
                kernel_size=1,
                stride=1,
            )
            for x in self.in_channels_list
        )

        self.grid = [torch.empty(0) for _ in range(self.num_detection_layer)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_detection_layer)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_detection_layer, -1, 2))

    def forward(self, x: List[Any]) -> Any:
        inference_outputs = []
        for i in range(self.num_detection_layer):
            x[i] = self.detector_head[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2: 4] != x[i].shape[2: 4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                inference_outputs.append(y.view(bs, self.num_anchors * nx * ny, self.num_outputs))

        return x if self.training else (torch.cat(inference_outputs, 1), x)

    def _make_grid(self, nx: int = 20, ny: int = 20, i: int = 0) -> Tuple[Any, Any]:
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.num_anchors, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), dim=2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.num_anchors, 1, 1, 2).expand(shape)
        return grid, anchor_grid

    def get_feat_index(self) -> List[int]:
        return self.in_channels_list

    def set_strides_anchors(self, all_strides: Dict[str, int]) -> None:
        self.stride = torch.tensor([v for k, v in all_strides.items() if int(k) in self.in_channels_list])
        self.anchors /= self.stride.view(-1, 1, 1)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.parser.add_argument(
            '--model.head.yolov5.anchors',
            nargs='+',
            type=int,
            default=(),
            help='Define anchor sizes as a flat list, e.g., --anchors 10 13 16 30 33 23'
        )
        parser.add_argument(
            '--model.head.yolov5.in_channels',
            nargs='+',
            type=int,
            default=(),
            help='The input channels of the YOLOv5 head.'
        )

        return parser




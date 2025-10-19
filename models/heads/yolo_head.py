"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.heads.base_head import BaseHead


class YOLOv5Head(BaseHead):
    """
    The head of YOLOv5.
    """
    def __init__(
            self,
            num_classes: int = 80,
            anchors: Tuple = (),
            in_channels: Optional[Tuple[int], List[int]] = (),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_detection_layer = len(anchors)
        self.num_anchors = len(anchors[0]) // 2

        self.detector_head = nn.ModuleList(
            nn.Conv2d(
                in_channels=x,
                out_channels=self.num_outputs * self.num_anchors,
                kernel_size=1,
                stride=1,
            )
            for x in in_channels
        )

        self.grid = [torch.empty(0) for _ in range(self.num_detection_layer)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_detection_layer)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_detection_layer, -1, 2))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        z = []




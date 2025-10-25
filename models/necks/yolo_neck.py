"""
Writen by: ian
"""
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

from models.components.yolo import CBS, C3
from models.necks.base_neck import BaseNeck

from models.necks import NECKS_REGISTRY


@NECKS_REGISTRY.register(component_name='yolov5_neck')
class YOLOv5Neck(BaseNeck):
    """
    The neck for YOLOv5
    """

    def __init__(
            self,
            opts: argparse.Namespace,
            in_channels_list: List[int],
            depth_multiple: float = 1,
    ) -> None:
        super().__init__(opts)

        assert len(in_channels_list) == 3, f"the length of input channel list must be 3, but got {len(in_channels_list)}"
        self.in_channels_list = in_channels_list
        c3_in, c4_in, c5_in = in_channels_list

        num_c3_blocks = max(round(3 * depth_multiple), 1)

        # --- FPN ---
        self.fpn_conv1 = CBS(c5_in, c4_in, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_c3_1 = C3(c4_in + c4_in, c4_in, num_blocks=num_c3_blocks, shortcut=False)

        self.fpn_conv2 = CBS(c4_in, c3_in, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_c3_2 = C3(c3_in + c3_in, c3_in, num_blocks=num_c3_blocks, shortcut=False)

        # --- PAN ---
        self.pan_conv1 = CBS(c3_in, c3_in, 3, 2)
        self.pan_c3_1 = C3(c3_in + c3_in, c4_in, num_blocks=num_c3_blocks, shortcut=False)

        self.pan_conv2 = CBS(c4_in, c4_in, 3, 2)
        self.pan_c3_2 = C3(c4_in + c4_in, c5_in, num_blocks=num_c3_blocks, shortcut=False)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (List[torch.Tensor]): The feature map list that obtain from backbone。
        """
        c3_feat, c4_feat, c5_feat = x

        # --- FPN ---
        fpn_out1 = self.fpn_conv1(c5_feat)

        fpn_out1_upsampled = self.upsample1(fpn_out1)
        fpn_out2 = torch.cat([fpn_out1_upsampled, c4_feat], dim=1)
        fpn_out2 = self.fpn_c3_1(fpn_out2)

        fpn_out2 = self.fpn_conv2(fpn_out2)
        fpn_out2_upsampled = self.upsample2(fpn_out2)
        fpn_out3 = torch.cat([fpn_out2_upsampled, c3_feat], dim=1)
        fpn_out3 = self.fpn_c3_2(fpn_out3)

        # --- PAN ---
        pan_out1 = self.pan_conv1(fpn_out3)
        pan_out1 = torch.cat([pan_out1, fpn_out2], dim=1)
        pan_out1 = self.pan_c3_1(pan_out1)

        pan_out2 = self.pan_conv2(pan_out1)
        pan_out2 = torch.cat([pan_out2, fpn_out1], dim=1)
        pan_out2 = self.pan_c3_2(pan_out2)

        return fpn_out3, pan_out1, pan_out2

    def inference(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Args:
                    x (List[torch.Tensor]): The feature map list that obtain from backbone。
                """
        c3_feat, c4_feat, c5_feat = x

        # --- FPN ---
        fpn_out1 = self.fpn_conv1.inference(c5_feat)

        fpn_out1_upsampled = self.upsample1(fpn_out1)
        fpn_out2 = torch.cat([fpn_out1_upsampled, c4_feat], dim=1)
        fpn_out2 = self.fpn_c3_1.inference(fpn_out2)

        fpn_out2 = self.fpn_conv2.inference(fpn_out2)
        fpn_out2_upsampled = self.upsample2(fpn_out2)
        fpn_out3 = torch.cat([fpn_out2_upsampled, c3_feat], dim=1)
        fpn_out3 = self.fpn_c3_2.inference(fpn_out3)

        # --- PAN ---
        pan_out1 = self.pan_conv1.inference(fpn_out3)
        pan_out1 = torch.cat([pan_out1, fpn_out2], dim=1)
        pan_out1 = self.pan_c3_1.inference(pan_out1)

        pan_out2 = self.pan_conv2.inference(pan_out1)
        pan_out2 = torch.cat([pan_out2, fpn_out1], dim=1)
        pan_out2 = self.pan_c3_2.inference(pan_out2)

        return fpn_out3, pan_out1, pan_out2

    def get_feat_index(self) -> List[int]:
        return self.in_channels_list

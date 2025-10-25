"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import List, Dict

import torch
from torch import Tensor

from models.architectures.base_architecture import BaseArchitecture

from utils.logger import logger
from utils.model_utils import summery_model, print_data
from models.architectures import ARCHITECTURES_REGISTRY
from models.backbones import build_backbone
from models.necks import build_neck
from models.heads import build_head


@ARCHITECTURES_REGISTRY.register(component_name='yolo', another_name='yolo_architecture')
class YOLO(BaseArchitecture):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)

        self.backbone = build_backbone(opts=opts)
        self.neck = build_neck(opts=opts)
        self.head = build_head(opts=opts)
        # TODO: the code have a problem, need to be fixed
        self.head.set_strides_anchors(self.backbone.get_strides())

        self.info()
        self.show_model_info()

    def backbone_forward(self, x) -> List[Tensor]:
        _ = self.backbone(x)
        outs = self.backbone.get_layer_out()
        return outs

    def forward(self, x) -> Tensor:
        features = self.backbone_forward(x)
        if self.neck is not None:
            features = self.neck(features)
        features = self.head(features)

        return features

    def show_model_info(self) -> None:
        logger.info("=" * 30 + "Model Summery" + "=" * 30)
        logger.info(f"Architecture: {self.__class__.__name__}")
        backbone_param, backbone_macs, backbone_flops = summery_model(
            model=self.backbone,
            image_size=self.input_size
        )
        print_data(
            title=f"Backbone: {self.backbone.__class__.__name__}",
            data={
                'param': backbone_param,
                'macs': backbone_macs,
                'flops': backbone_flops
            },
            unit={
                'param': 'M',
                'macs': 'G',
                'flops': 'G'
            }
        )
        if self.neck is not None:
            feature_map_size = self.backbone.get_map_size(input_size=self.input_size)
            feature_index = self.neck.get_feat_index()
            neck_inputs = self.create_virtual_input(all_feat_size=feature_map_size, feat_index=feature_index)
            neck_param, neck_macs, neck_flops = summery_model(
                model=self.neck,
                input_virtual=neck_inputs,
            )
            print_data(
                title=f"Neck: {self.neck.__class__.__name__}",
                data={
                    'param': neck_param,
                    'macs': neck_macs,
                    'flops': neck_flops
                },
                unit={
                    'param': 'M',
                    'macs': 'G',
                    'flops': 'G'
                }
            )
        else:
            neck_param, neck_macs, neck_flops = 0, 0, 0
        if self.head is not None:
            feature_map_size = self.backbone.get_map_size(input_size=self.input_size)
            feature_index = self.head.get_feat_index()
            head_inputs = self.create_virtual_input(all_feat_size=feature_map_size, feat_index=feature_index)
            head_param, head_macs, head_flops = summery_model(
                model=self.head,
                input_virtual=head_inputs,
            )
            print_data(
                title=f"Head: {self.head.__class__.__name__}",
                data={
                    'param': head_param,
                    'macs': head_macs,
                    'flops': head_flops
                },
                unit={
                    'param': 'M',
                    'macs': 'G',
                    'flops': 'G'
                }
            )
        else:
            head_param, head_macs, head_flops = 0, 0, 0
        total_param = backbone_param + neck_param + head_param
        total_macs = backbone_macs + neck_macs + head_macs
        total_flops = backbone_flops + neck_flops + head_flops
        print_data(
            title="Total:",
            data={
                'param': total_param,
                'macs': total_macs,
                'flops': total_flops
            },
            unit={
                'param': 'M',
                'macs': 'G',
                'flops': 'G'
            }
        )

    def create_virtual_input(
            self,
            all_feat_size: Dict[str, int],
            feat_index: List[int]
    ) -> List[Tensor]:
        virtual_inputs = []
        for index in feat_index:
            feat_size = all_feat_size[str(index)]
            virtual_input = torch.empty((1, index, feat_size, feat_size), device=self.device)
            virtual_inputs.append(virtual_input)

        return virtual_inputs

    def info(self) -> None:
        logger.info("=" * 75)
        logger.info("Loading model...")
        logger.info(f"Load Backbone: {self.backbone.__class__.__name__}")
        if self.neck is not None:
            logger.info(f"Load Neck: {self.neck.__class__.__name__}")
        logger.info(f"Load Head: {self.head.__class__.__name__}")
        logger.info(f"The model will be deployed on {self.device.upper()}")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)

        return parser


if __name__ == "__main__":
    model = YOLO()


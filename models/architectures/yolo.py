"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Optional, Union, Tuple, List, Dict

import torch
from torch import Tensor

from models.architectures.base_architecture import BaseArchitecture
from models.backbones.base_backbone import BaseBackbone
from models.necks.base_neck import BaseNeck
from models.heads.base_head import BaseHead

from utils.logger import logger
from utils.model_utils import summery_model, print_data
from models.architectures import ARCHITECTURES_REGISTRY
from models.backbones import build_backbone
from models.necks import build_neck


@ARCHITECTURES_REGISTRY.register(component_name='yolo', another_name='yolo_architecture')
class YOLO(BaseArchitecture):
    def __init__(
            self,
            backbone: BaseBackbone = None,
            neck: BaseNeck = None,
            head: BaseHead = None,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = 640,
            device: str = None,
    ) -> None:
        super().__init__(input_size=input_size, device=device)

        if backbone is None:
            logger.warning("Backbone is not provided, use the Yolo version 5 as default."
                           "If you do not want to use yolov5, please set provide your own "
                           "model base on yolo architecture.")
            self.backbone = build_backbone(
                backbone_name="csp_darknet53",
                args={"width_multiple": 1.0, "depth_multiple": 1.0}
            )
            self.neck = build_neck(
                neck_name="yolov5_neck",
                args={
                    "in_channels_list": [256, 512, 1024],
                    "depth_multiple": 1.0
                }
            )
            self.head = head
        else:
            self.backbone = backbone
            self.neck = neck
            self.head = head

        self.info()
        self.show_model_info()

    def backbone_forward(self, x) -> Tensor:
        self.backbone(x)

        return x

    def forward(self, x) -> Tensor:
        pass

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
            head_param, head_macs, head_flops = summery_model(
                model=self.neck,
                image_size=self.input_size
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
        logger.info("Loading custom model...")
        logger.info(f"Load Backbone: {self.backbone.__class__.__name__}")
        if self.neck is not None:
            logger.info(f"Load Neck: {self.neck.__class__.__name__}")
        logger.info(f"Load Head: {self.head.__class__.__name__}")
        logger.info(f"The model will be deployed on {self.device.upper()}")



if __name__ == "__main__":
    model = YOLO()


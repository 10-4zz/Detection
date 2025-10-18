"""
Writen by: ian
"""
from typing import Optional, Union, Tuple, List
from torch import Tensor

from models.architectures.base_architecture import BaseArchitecture
from models.backbones.base_backbone import BaseBackbone
from models.necks.base_neck import BaseNeck
from models.heads.base_head import BaseHead

from utils.logger import logger
from utils.model_utils import summery_model, print_data
from models.architectures import ARCHITECTURES_REGISTRY
from models.backbones import build_backbone


@ARCHITECTURES_REGISTRY.register(component_name='yolo', another_name='yolo_architecture')
class YOLO(BaseArchitecture):
    def __init__(
            self,
            backbone: BaseBackbone = None,
            neck: BaseNeck = None,
            head: BaseHead = None,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = 640,
    ) -> None:
        super().__init__(input_size=input_size)

        if backbone is None:
            logger.warning("Yolo version and backbone is not provided, use the Yolo version 5 as default."
                           "If you do not want to use yolov5, please set the version or provide your own "
                           "model base on yolo architecture.")
            self.backbone = build_backbone(
                backbone_name="csp_darknet53",
                args={"width_multiple": 1.0, "depth_multiple": 1.0}
            )
            self.neck = neck
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
        logger.info("========================Model Summery==========================")
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
            neck_param, neck_macs, neck_flops = summery_model(
                model=self.neck,
                image_size=self.input_size
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
        logger.info("===============================================================")

    def info(self) -> None:
        logger.info("========================================================")
        logger.info("Loading custom model...")
        logger.info(f"Load Backbone: {self.backbone.__class__.__name__}")
        if self.neck is not None:
            logger.info(f"Load Neck: {self.neck.__class__.__name__}")
        logger.info(f"Load Head: {self.head.__class__.__name__}")



if __name__ == "__main__":
    model = YOLO()


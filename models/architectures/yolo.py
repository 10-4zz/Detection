"""
Writen by: ian
"""
from torch import Tensor

from models.architectures.base_architecture import BaseArchitecture
from models.backbones.base_backbone import BaseBackbone
from models.necks.base_neck import BaseNeck
from models.heads.base_head import BaseHead

from utils.logger import logger
from models.architectures import ARCHITECTURES_REGISTRY


@ARCHITECTURES_REGISTRY.register(component_name='yolo', another_name='yolo_architecture')
class YOLO(BaseArchitecture):
    def __init__(
            self,
            version: str = None,
            backbone: BaseBackbone = None,
            neck: BaseNeck = None,
            head: BaseHead = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

        if version is None:
            if self.backbone is None:
                logger.warning("Yolo version and backbone is not provided, use the Yolo version 5 as default."
                               "If you do not want to use yolov5, please set the version or provide your own "
                               "model base on yolo architecture.")
            else:
                pass
        else:
            pass

        self.info()

    def backbone_forward(self, x) -> Tensor:
        self.backbone(x)

        return x

    def forward(self, x) -> Tensor:
        pass

    def show_model_info(self) -> None:
        pass

    def info(self) -> None:
        logger.info("========================================================")
        logger.info("Loading custom model...")
        logger.info(f"Load Backbone: {self.backbone.__class__.__name__}")
        if self.neck is not None:
            logger.info(f"Load Neck: {self.neck.__class__.__name__}")
        logger.info(f"Load Head: {self.head.__class__.__name__}")
        logger.info("========================================================")



if __name__ == "__main__":
    model = YOLO()


"""
Writen by: ian
"""
from typing import Optional, Union


from models.architectures.base_architecture import BaseArchitecture
from models.backbones.base_backbone import BaseBackbone
from models.necks.base_neck import BaseNeck
from models.heads.base_head import BaseHead

from utils.logger import logger


class YOLO(BaseArchitecture):
    def __init__(
            self,
            version: str = None,
            model_scale: Optional[Union[int, float, str]] = 1.0,
            backbone: BaseBackbone = None,
            neck: BaseNeck = None,
            head: BaseHead = None,
    ):
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
                logger.info("Loading custom model...")
                logger.info(f"Load Backbone: {self.backbone.__class__.__name__}")
                if self.neck is not None:
                    logger.info(f"Load Neck: {self.neck.__class__.__name__}")
                logger.info(f"Load Head: {self.head.__class__.__name__}")
        else:
            pass


if __name__ == "__main__":
    model = YOLO()


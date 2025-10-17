"""
Writen by: ian
"""
from models.backbones.base_backbone import BaseBackbone
from models.backbones import BACKBONES_REGISTRY





@BACKBONES_REGISTRY.register(component_name="csp_darknet53")
class CSPDarkNet53(BaseBackbone):
    def __init__(self) -> None:
        super().__init__()


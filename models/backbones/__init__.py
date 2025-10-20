"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Dict, Any

import torch.nn as nn

from utils.registry import Registry

BACKBONES_REGISTRY = Registry(
    registry_name="Backbone_Registry",
    component_dir=["models/backbones"],
)


def build_backbone(backbone_name: str, args: Dict[str, Any]) -> nn.Module:
    create_fn = BACKBONES_REGISTRY.get(backbone_name)
    backbone = create_fn(**args)
    return backbone

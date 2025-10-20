"""
Writen by: ian
"""
from typing import Dict, Any

import torch.nn as nn

from utils.registry import Registry

NECKS_REGISTRY = Registry(
    registry_name="Neck_Registry",
    component_dir=["models/necks"],
)


def build_neck(neck_name: str, args: Dict[str, Any]) -> nn.Module:
    create_fn = NECKS_REGISTRY.get(neck_name)
    neck = create_fn(**args)
    return neck

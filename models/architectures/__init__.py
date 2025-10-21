"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Dict, Any

import torch.nn as nn

from utils.registry import Registry

ARCHITECTURES_REGISTRY = Registry(
    registry_name="Architecture_Registry",
    component_dir=["models/architectures"],
)


def build_architecture(architecture_name: str, args: Dict[str, Any]) -> nn.Module:
    """
    Build a architecture.
    """
    create_fn = ARCHITECTURES_REGISTRY.get(architecture_name)
    architecture = create_fn(**args)
    return architecture


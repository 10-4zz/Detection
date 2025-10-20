"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Dict, Any

import torch.nn as nn

from utils.registry import Registry

HEADS_REGISTRY = Registry(
    registry_name="Head_Registry",
    component_dir=["models/heads"],
)


def build_head(head_name: str, args: Dict[str, Any]) -> nn.Module:
    create_fn = HEADS_REGISTRY.get(head_name)
    head = create_fn(**args)
    return head



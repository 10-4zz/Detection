"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Dict, Any

import torch.nn as nn

from models.backbones.base_backbone import BaseBackbone
from utils.registry import Registry

BACKBONES_REGISTRY = Registry(
    registry_name="Backbone_Registry",
    component_dir=["models/backbones"],
)


def build_backbone(backbone_name: str, args: Dict[str, Any]) -> nn.Module:
    create_fn = BACKBONES_REGISTRY.get(backbone_name)
    backbone = create_fn(**args)
    return backbone


def arguments_backbone(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add backbone specific arguments to the parser.
    """
    parser = BaseBackbone.add_arguments(parser=parser)
    parser = BACKBONES_REGISTRY.all_arguments(parser=parser)
    return parser

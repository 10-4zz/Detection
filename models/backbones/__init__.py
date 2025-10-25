"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse

import torch.nn as nn

from models.backbones.base_backbone import BaseBackbone
from utils.registry import Registry

BACKBONES_REGISTRY = Registry(
    registry_name="Backbone_Registry",
    component_dir=["models/backbones"],
)


def build_backbone(opts: argparse.Namespace) -> nn.Module:
    backbone_name = getattr(opts, "model.backbone.name", None)
    create_fn = BACKBONES_REGISTRY.get(backbone_name)
    backbone = create_fn(opts)
    return backbone


def arguments_backbone(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add backbone specific arguments to the parser.
    """
    parser = BaseBackbone.add_arguments(parser=parser)
    parser = BACKBONES_REGISTRY.all_arguments(parser=parser)
    return parser

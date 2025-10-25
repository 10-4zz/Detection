"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Dict, Any

import torch.nn as nn

from models.architectures.base_architecture import BaseArchitecture
from utils.registry import Registry

ARCHITECTURES_REGISTRY = Registry(
    registry_name="Architecture_Registry",
    component_dir=["models/architectures"],
)


def build_architecture(opts: argparse.Namespace) -> nn.Module:
    """
    Build an architecture.
    """
    architecture_name = getattr(opts, "model.architecture.name", "yolo")
    create_fn = ARCHITECTURES_REGISTRY.get(architecture_name)
    architecture = create_fn(opts)
    return architecture


def arguments_architecture(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add architecture-specific arguments to the parser.
    """
    parser = BaseArchitecture.add_arguments(parser=parser)
    parser = ARCHITECTURES_REGISTRY.all_arguments(parser=parser)
    return parser

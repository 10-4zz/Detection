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


def build_architecture(architecture_name: str, args: Dict[str, Any]) -> nn.Module:
    """
    Build an architecture.
    """
    create_fn = ARCHITECTURES_REGISTRY.get(architecture_name)
    architecture = create_fn(**args)
    return architecture


def arguments_architecture(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add architecture-specific arguments to the parser.
    """
    parser = BaseArchitecture.add_arguments(parser=parser)
    parser = ARCHITECTURES_REGISTRY.all_arguments(parser=parser)
    return parser

"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Dict, Any

import torch.nn as nn

from models.heads.base_head import BaseHead
from utils.registry import Registry

HEADS_REGISTRY = Registry(
    registry_name="Head_Registry",
    component_dir=["models/heads"],
)


def build_head(opts: argparse.Namespace) -> nn.Module:
    head_name = getattr(opts, "model.head.name", None)
    create_fn = HEADS_REGISTRY.get(head_name)
    head = create_fn(opts)
    return head


def arguments_heads(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for heads.
    :param parser:
    :return:
    """
    parser = BaseHead.add_arguments(parser=parser)
    parser = HEADS_REGISTRY.all_arguments(parser=parser)
    return parser

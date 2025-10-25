"""
Writen by: ian
"""
import argparse

import torch.nn as nn

from models.necks.base_neck import BaseNeck
from utils.registry import Registry

NECKS_REGISTRY = Registry(
    registry_name="Neck_Registry",
    component_dir=["models/necks"],
)


def build_neck(opts: argparse.Namespace) -> nn.Module:
    neck_name = getattr(opts, "model.neck.name", None)
    create_fn = NECKS_REGISTRY.get(neck_name)
    neck = create_fn(opts)
    return neck


def arguments_necks(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for necks.
    :param parser:
    :return:
    """
    parser = BaseNeck.add_arguments(parser=parser)
    parser = NECKS_REGISTRY.all_arguments(parser=parser)
    return parser

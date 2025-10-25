"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse

import torch.nn as nn

from models.architectures import build_architecture, arguments_architecture
from models.backbones import arguments_backbone
from models.heads import arguments_heads
from models.necks import arguments_necks


def build_model(opts: argparse.Namespace) -> nn.Module:
    """
    Build a model.
    """
    model = build_architecture(opts=opts)

    return model



def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for model.
    :param parser:
    :return:
    """
    parser = arguments_architecture(parser=parser)
    parser = arguments_backbone(parser=parser)
    parser = arguments_necks(parser=parser)
    parser = arguments_heads(parser=parser)
    return parser


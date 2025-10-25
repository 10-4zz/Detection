"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Dict, Any

import torch.nn as nn

from models.architectures import build_architecture
from models.backbones import build_backbone
from models.heads import build_head
from models.necks import build_neck


def build_model(opts: argparse.Namespace) -> nn.Module:
    """
    Build a model.
    """
    backbone = build_backbone(
        backbone_name=components_name['backbone'],
        args=components_args['backbone']
    )
    if components_name['neck'] is not None:
        neck = build_neck(
            neck_name=components_name['neck'],
            args=components_args['neck']
        )
    else:
        neck = None
    head = build_head(
        head_name=components_name['head'],
        args=components_args['head']
    )
    model_args = {
        "backbone": backbone,
        "neck": neck,
        "head": head
    }
    model = build_architecture(
        architecture_name=components_name['architecture'],
        args=model_args
    )
    return model


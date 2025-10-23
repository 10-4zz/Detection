"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse

import torch.nn as nn

from models.base import Base


class BaseHead(Base):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)

    def forward(self, x):
        raise NotImplementedError

    def get_feat_index(self):
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.head.name", type=str, default=None, help="The name of the head.")
        group.add_argument("--model.head.num_classes", type=int, default=80, help="The number of classes.")

        return parser


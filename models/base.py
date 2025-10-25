"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse

import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__()

        device = getattr(opts, "device", None)
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = getattr(opts, "model.input_size", 640)
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.input_size", type=int, default=640, help="Input size of the model")

        return parser

"""
For licensing see accompanying LICENSE file.
Writen by ian
"""
import argparse

import torch
import torch.nn as nn

from models.base import Base
from utils.logger import logger


class BaseArchitecture(Base):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)


    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")

    def show_model_info(self) -> None:
        """
        This method will provide the information about the architecture or model. for example, parameters
        FLOPs and so on.
        """
        logger.warning(
            f"If you want to learn about the information about {self.__class__.__name__}'s parameters, FLOPS and so on, "
            "please implement the method get_model_info in your class."
        )

    def info(self) -> None:
        """
        This method is used to show some information which need be shown in terminal.
        :return:
        """
        pass

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.architecture.name", type=str, default=None, help="The name of the architecture.")

        return parser




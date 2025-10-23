"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import List, Tuple, Union, Optional, Dict

from torch import Tensor

from models.base import Base
from utils.logger import logger


class BaseBackbone(Base):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)

        self.out_index = getattr(opts, "model.backbone.output_index", None)
        self.layer_out: List[Tensor] = []

        self.in_proj = None

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None

        self.out_proj = None
        self.opts = opts

    def forward(self, x) -> Tensor:
        if self.in_proj is not None:
            x = self.in_proj(x)
            self.layer_out.append(x)

        if self.layer1 is not None:
            x = self.layer1(x)
            self.layer_out.append(x)

        if self.layer2 is not None:
            x = self.layer2(x)
            self.layer_out.append(x)

        if self.layer3 is not None:
            x = self.layer3(x)
            self.layer_out.append(x)

        if self.layer4 is not None:
            x = self.layer4(x)
            self.layer_out.append(x)

        if self.layer5 is not None:
            x = self.layer5(x)
            self.layer_out.append(x)

        if self.out_proj is not None:
            x = self.out_proj(x)
            self.layer_out.append(x)

        return x

    def get_layer_out(self) -> List[Tensor]:
        if self.out_index is None:
            logger.warning("No index provided, returning final outputs.")
            return [self.layer_out[-1]]
        return [self.layer_out[i] for i in self.out_index]

    def get_map_size(
            self,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = 640,
    ) -> Dict[str, int]:
        raise NotImplementedError("Please Implement get_map_size method")

    def get_strides(self):
        raise NotImplementedError("Please Implement get_strides method")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.backbone.name", type=str, default=None, help="The name of the backbone.")
        group.add_argument(
            '--model.backbone.out_index',
            type=int,
            nargs='+',
            default=None,
            help='The out index of the backbone.'
        )

        return parser
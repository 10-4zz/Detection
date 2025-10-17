"""
Writen by: ian
"""
from typing import List, Tuple, Union, Optional

import torch.nn as nn

from torch import Tensor

from utils.logger import logger


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_out: List[Tensor] = []

        self.in_proj = None

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None

        self.out_proj = None

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

    def get_layer_out(self, index: Optional[Union[List[int], Tuple[int]]] = None) -> List[Tensor]:
        if index is None:
            logger.warning("No index provided, returning all layer outputs.")
            return self.layer_out

        return [self.layer_out[i] for i in index]

"""
Writen by ian
"""
from typing import Optional, Union, Tuple, List

import torch.nn as nn
from torch import Tensor

from utils.logger import logger


class BaseArchitecture(nn.Module):
    def __init__(
            self,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = None,
    ) -> None:
        super(BaseArchitecture, self).__init__()

        self.input_size = input_size
        self.show_model_info()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")

    def show_model_info(self) -> None:
        """
        This method will provide the information about the architecture or model. for example, parameters
        FLOPs and so on.
        """
        logger.warning(
            f"If you want to learn about the information about {self.__class__.__name__}'s parameters, FLOPS eta, "
            "please implement the method get_model_info in your class."
        )

    def info(self) -> None:
        """
        This method is used to show some information which need be shown in terminal.
        :return:
        """
        pass




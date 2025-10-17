"""
Writen by ian
"""
from typing import Dict, Optional, Union, Tuple, List

import torch.nn as nn

from utils.logger import logger


class BaseArchitecture(nn.Module):
    def __init__(
            self,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = None,
    ):
        super(BaseArchitecture, self).__init__()

        self.input_size = input_size
        self.get_model_info()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")

    def get_model_info(self):
        """
        This method will provide the information about the architecture or model. for example, parameters
        FLOPs and so on.
        """
        logger.warning(
            f"If you want to learn about the information about {self.__class__.__name__}, "
            "please implement the method get_model_info in your class."
        )




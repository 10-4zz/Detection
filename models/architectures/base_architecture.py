"""
Writen by ian
"""
from typing import Dict, Optional, Union, Tuple, List

import torch.nn as nn


class BaseArchitecture(nn.Module):
    def __init__(
            self,
            input_size: Optional[Union[int, Tuple[int], List[list]]] = None,
    ):
        super(BaseArchitecture, self).__init__()
        self.input_size = input_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")

    def get_model_info(self) -> Dict:
        """
        This method will provide the information about the architecture or model. for example, parameters
        FLOPs and so on.
        :return:
            model_info: str
        """
        pass



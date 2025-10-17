from copy import deepcopy
from typing import Any, Tuple
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.logger import logger

try:
    import thop
    THOP_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    THOP_AVAILABLE = False
    print("Could not import thop. This is OK if you do not need to obtain the macs or flops of model.")



def summery_model(
        model: nn.Module,
        input_virtual: Any = None,
        image_size: int = 224,
) -> Tuple[int, float, Any]:
    """
    Args:
        model (torch.nn.Module): The model to summarize.
        input_virtual (Any, optional): A virtual input tensor for the model. Defaults to None.
        image_size (int, optional): The size of the input image. Defaults to 224.
    """

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if THOP_AVAILABLE:
        p = next(model.parameters())
        if input_virtual is not None:
            im = input_virtual
        else:
            im = torch.empty((1, 3, image_size, image_size), device=p.device)

        macs, _ = thop.profile(deepcopy(model), inputs=(im,), verbose=False)
        flops = macs * 2
    else:
        logger.warning(
            "The `thop` library is not installed. "
            "MACs and FLOPs calculation will be skipped. "
            "To enable this, please install thop: `pip install thop`"
        )
        macs, flops = None, None

    return n_parameters, macs, flops


def throughput(
        data_loader: DataLoader,
        model: nn.Module,
):
    """
    Args:
        data_loader (torch.utils.data.DataLoader): The data loader for the model.
        model (torch.nn.Module): The model to evaluate throughput.
    """
    with torch.no_grad():
        model.eval()

        for _, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            tic1 = time.time()
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            return
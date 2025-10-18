from copy import deepcopy
from typing import Any, Tuple, Dict, Optional
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


def print_data(title: str, data: Dict[str, Optional[float]], unit: Dict[str, str]) -> None:
    """
    print the data of model.

    Args:
        title (str): 打印信息的标题。
        data (Dict[str, Optional[float]]): 包含原始数据的字典, e.g., {'param': 11.7e6, 'macs': 5.5e9, 'flops': None}
        unit (Dict[str, str]): 包含每个数据显示单位的字典, e.g., {'param': 'M', 'macs': 'G', 'flops': 'G'}
    """
    logger.info(title)

    divisors = {'K': 1e3, 'M': 1e6, 'G': 1e9}

    metrics_to_print = {
        'param': {"name": "Parameters", "format": ".3f"},
        'macs': {"name": "MACs (Thop)", "format": ".1f"},
        'flops': {"name": "FLOPs (Thop)", "format": ".1f"},
    }

    for key, props in metrics_to_print.items():
        raw_value = data.get(key)
        target_unit = unit.get(key, "")

        if raw_value is None:
            display_name = props["name"]
            logger.info(f"{display_name:<15}= {'N/A (thop dependency missing)'}")
            continue

        divisor = divisors.get(target_unit, 1)
        scaled_value = raw_value / divisor

        display_name = props["name"]
        number_format = props["format"]

        logger.info(f"{display_name:<15}= {scaled_value:8{number_format}}{target_unit}")


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
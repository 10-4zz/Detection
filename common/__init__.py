"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Tuple

import random
import numpy as np
import torch

from utils.logger import logger


def set_random_seed(opts: argparse.Namespace) -> None:
    """
    Set random seed for reproducibility.
    :param opts:
    :return:
    """
    logger.info("=" * 20 + " Setting Random Seed " + "=" * 20)
    seed = getattr(opts, "common.random_seed", 42)
    deterministic = getattr(opts, "cudnn.deterministic", True)
    benchmark = getattr(opts, "cudnn.benchmark", False)
    logger.info("Set random seed to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic, benchmark = check_deterministic_and_benchmark(deterministic, benchmark)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    logger.info("cudnn.deterministic set to {}, cudnn.benchmark set to {}".format(deterministic, benchmark))
    if deterministic and not benchmark:
        logger.info("Note: This setting may lead to slow training speed, but ensures reproducibility.")
    else:
        logger.info("Note: This setting may lead to non-deterministic results, but improves training speed.")
    logger.info("Random seed set successfully.")


def check_deterministic_and_benchmark(deterministic: bool, benchmark: bool) -> Tuple[bool, bool]:
    """
    Check the current settings for cudnn deterministic and benchmark.
    :return:
    """
    if deterministic:
        if benchmark:
            logger.warning("cudnn.deterministic is True, but cudnn.benchmark is also True. "
                           "This may lead to non-deterministic results. The cudnn.benchmark will be set to False.")
            benchmark = False
    else:
        if not benchmark:
            logger.warning("cudnn.deterministic is False, and cudnn.benchmark is also False. "
                           "This may lead to slow training. The cudnn.benchmark will be set cudnn.benchmark"
                           " to True for better performance.")
            benchmark = True

    return deterministic, benchmark


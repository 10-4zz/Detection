"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import sys
import os
import torch
import platform

from utils.logger import logger

def log_environment_info():
    """
    get and log thr information of environment.
    """
    logger.info("=" * 25 + " Environment Information " + "=" * 25)

    logger.info(f"{'Operating System':<25}: {platform.system()} {platform.release()}")

    logger.info(f"{'Python Version':<25}: {sys.version.split(' ')[0]}")

    logger.info(f"{'PyTorch Version':<25}: {torch.__version__}")

    cpu_count = os.cpu_count()
    logger.info(f"{'CPU Count':<25}: {cpu_count}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"{'CUDA Available':<25}: True")
        logger.info(f"{'GPU Count':<25}: {gpu_count}")

        logger.info(f"{'PyTorch Built with CUDA':<25}: {torch.version.cuda}")

        if torch.backends.cudnn.is_available():
            logger.info(f"{'cuDNN Version':<25}: {torch.backends.cudnn.version()}")
        else:
            logger.info(f"{'cuDNN Version':<25}: Not Available")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"  - GPU {i}: {gpu_name}, Total Memory: {total_mem:.2f} GB")
    else:
        logger.info(f"{'CUDA Available':<25}: False (PyTorch is in CPU-only mode)")



if __name__ == '__main__':
    log_environment_info()
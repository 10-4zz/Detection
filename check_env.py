"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from utils.env_utils import check_dependencies, check_cuda_torch_compatibility
from utils.logger import logger


def main():
    logger.info("=" * 70)
    check_dependencies()
    logger.info("=" * 70)
    check_cuda_torch_compatibility()
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

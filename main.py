"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from models.architectures.yolo import YOLO
from utils.logger import logger
from utils.env_utils import log_environment_info


def main():
    log_environment_info()
    model = YOLO(device='cpu')


if __name__ == "__main__":
    main()



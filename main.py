"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from models import build_model
from utils.logger import logger
from utils.env_utils import log_environment_info
from utils.opts import get_base_args


def main():
    # get args
    args = get_base_args().parse_args()
    log_environment_info()
    model = build_model()


if __name__ == "__main__":
    main()



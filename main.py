"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from models import build_model
from utils.logger import logger
from utils.env_utils import log_environment_info
from utils.opts import get_train_args


def main():
    # get args
    opts = get_train_args()
    log_environment_info()
    model = build_model(opts=opts)


if __name__ == "__main__":
    main()



"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""

from common import set_random_seed
from data import build_dataloader
from models import build_model
from utils.logger import logger
from utils.env_utils import log_environment_info
from utils.opts import get_train_args


def main():
    # get args
    opts = get_train_args()
    set_random_seed(opts)
    log_environment_info()
    train_loader, val_loader = build_dataloader(opts)
    model = build_model(opts=opts)


if __name__ == "__main__":
    main()



"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import os
import sys
import logging
import copy
import functools
from datetime import datetime
from termcolor import colored


class CustomColorFormatter(logging.Formatter):
    """
    A custom logging formatter that adds colors to log levels.
    """
    LEVEL_COLORS = {
        logging.DEBUG: 'grey',
        logging.INFO: 'blue',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red',
    }

    def format(self, record):
        record_copy = copy.deepcopy(record)

        color = self.LEVEL_COLORS.get(record_copy.levelno, 'white')

        record_copy.levelname = colored(record_copy.levelname, color)

        return super().format(record_copy)


@functools.lru_cache()
def create_logger(output_dir, filename=None, name=''):
    # create logger
    os.makedirs(output_dir, exist_ok=True)
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s]: %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + ': %(levelname)s %(message)s'
    console_formatter = CustomColorFormatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}"

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt=console_formatter)
    _logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(output_dir, f'{filename}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(file_handler)

    return _logger


logger = create_logger(
    os.environ.get("LOG_DIR", "./output/log"),
    os.environ.get("LOG_FILE_NAME", None),
    os.environ.get("LOG_NAME", "detection")
)


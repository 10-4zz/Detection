"""
Writen by: ian
"""
import os
import sys
import logging
import functools
from datetime import datetime
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, filename=None, name=''):
    # create logger
    os.makedirs(output_dir, exist_ok=True)
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}"

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
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


"""
This file content from ml-cvnets that is created by Apple in most part.
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
import collections
import re
import os

import yaml

from data.datasets import arguments_datasets
from models import arguments_model

from utils.ddp_utils import is_master
from utils.logger import logger

try:
    # Workaround for DeprecationWarning when importing Collections
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

DEFAULT_CONFIG_DIR = "config"
META_PARAMS_REGEX = ""


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(opts):
    config_file_name = getattr(opts, "config", None)
    if config_file_name is None:
        return opts
    # This is used to distribute training.
    is_master_node = is_master(opts)

    # if is_master_node:
    #     config_file_name = get_local_path(opts=opts, path=config_file_name)

    if not os.path.isfile(config_file_name):
        if len(config_file_name.split("/")) == 1:
            # loading files from default config folder
            new_config_file_name = "{}/{}".format(DEFAULT_CONFIG_DIR, config_file_name)
            if not os.path.isfile(new_config_file_name) and is_master_node:
                logger.error(
                    "Configuration file neither exists at {} nor at {}".format(
                        config_file_name, new_config_file_name
                    )
                )
            else:
                config_file_name = new_config_file_name
        else:
            # If absolute path of the file is passed
            if not os.path.isfile(config_file_name) and is_master_node:
                logger.error(
                    "Configuration file does not exists at {}".format(config_file_name)
                )

    setattr(opts, "common.config_file", config_file_name)
    with open(config_file_name, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
                elif "local_" not in k and not re.match(META_PARAMS_REGEX, k):
                    message = (
                        f"Yaml config key '{k}' was not recognized by argparser. If you think that you have already added "
                        f"argument in options/opts.py file, then check for typos. If not, then please add it to options/opts.py."
                    )
                    logger.warning(message)
        except yaml.YAMLError as exc:
            if is_master_node:
                logger.error(
                    "Error while loading config file: {}. Error message: {}".format(
                        config_file_name, str(exc)
                    )
                )

    # override arguments
    override_args = getattr(opts, "override_args", None)
    if override_args is not None:
        for override_k, override_v in override_args.items():
            if hasattr(opts, override_k):
                setattr(opts, override_k, override_v)
            elif "local_" not in k and not re.match(META_PARAMS_REGEX, k):
                message = (
                    f"Yaml config key '{override_k}' was not recognized by argparser. If you think that you have already added "
                    f"argument in options/opts.py file, then check for typos. If not, then please add it to options/opts.py."
                )
                logger.warning(message)

    return opts


# def extend_selected_args_with_prefix(
#     parser: argparse.ArgumentParser, match_prefix: str, additional_prefix: str
# ) -> argparse.ArgumentParser:
#     """
#     Helper function to select arguments with certain prefix and duplicate them with a replaced prefix.
#     An example use case is distillation, where we want to add --teacher.model.* as a prefix to all --model.* arguments.
#
#     In that case, we provide the following arguments:
#     * match_prefix="--model."
#     * additional_prefix="--teacher.model."
#
#     Args:
#         parser: The argument parser to extend.
#         match_prefix: Prefix to select arguments for duplication.
#             The value should start with "--", contain no underscores, and with ".".
#         additional_prefix: Prefix to replace the @match_prefix in duplicated arguments.
#             The value should start with "--", contain no underscores, and with ".".
#     """
#     # all arguments are stored as actions
#     options = parser._actions
#
#     regexp = r"--[^_]+\."
#     assert re.match(
#         regexp, match_prefix
#     ), f"match prefix '{match_prefix}' should match regexp '{regexp}'"
#     assert re.match(
#         regexp, additional_prefix
#     ), f"additional prefix '{additional_prefix}' should match regexp '{regexp}'"
#
#     for option in options:
#         option_strings = option.option_strings
#         # option strings are stored as a list
#         for option_string in option_strings:
#             if option_string.startswith(match_prefix):
#                 parser.add_argument(
#                     option_string.replace(match_prefix, additional_prefix),
#                     nargs="?"
#                     if isinstance(option, argparse._StoreTrueAction)
#                     else option.nargs,
#                     const=option.const,
#                     default=option.default,
#                     type=option.type,
#                     choices=option.choices,
#                     help=option.help,
#                     metavar=option.metavar,
#                 )
#     return parser
#
#
# def extract_opts_with_prefix_replacement(
#     opts: argparse.Namespace,
#     match_prefix: str,
#     replacement_prefix: str,
# ) -> argparse.Namespace:
#     """
#     Helper function to extract a copy options with certain prefix and return them with an alternative prefix.
#     An example usage is distillation, when we have used @extend_selected_args_with_prefix to add --teacher.model.*
#         arguments to argparser, and now we want to re-use the handlers of model.* opts by teacher.model.* opts
#
#     Args:
#         opts: The argument parser to extend.
#         match_prefix: Prefix to select opts for extraction.
#             The value should not contain dashes and should end with "."
#         replacement_prefix: Prefix to replace the @match_prefix
#             The value should not contain dashes and should end with "."
#     """
#     regexp = r"[^-]+\."
#     assert re.match(
#         regexp, match_prefix
#     ), f"match prefix '{match_prefix}' should match regexp '{regexp}'"
#     assert re.match(
#         regexp, replacement_prefix
#     ), f"replacement prefix '{replacement_prefix}' should match regexp '{regexp}'"
#
#     opts_dict = vars(opts)
#     result_dict = {
#         # replace teacher with empty string in "teacher.model.*" to get model.*
#         key.replace(match_prefix, replacement_prefix): value
#         for key, value in opts_dict.items()
#         # filter keys related to teacher
#         if key.startswith(match_prefix)
#     }
#
#     return argparse.Namespace(**result_dict)


def common_args():
    parser = argparse.ArgumentParser(description="Detection Model Arguments")

    # Add basic arguments
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--common.seed', type=int, default=42, help='Random seed')
    parser.add_argument('--common.device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--common.output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--common.tag', type=str, default='', help='Tag for the experiment')

    return parser


def cudnn_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for cudnn.
    :param parser:
    :return:
    """
    group = parser.add_argument_group(title='Arguments for cudnn.')

    # Add cudnn specific arguments
    group.add_argument('--cudnn.deterministic', type=bool, default=True, help='Whether to set cudnn.deterministic to True')
    group.add_argument('--cudnn.benchmark', type=bool, default=False, help='Whether to set cudnn.benchmark to True')

    return parser


def data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for data.
    :param parser:
    :return:
    """
    parser = arguments_datasets(parser=parser)

    group = parser.add_argument_group(title='Arguments for dataset.')

    # Add data specific arguments
    group.add_argument('--data.num_workers', type=int, default=4, help='Number of data loading workers')
    group.add_argument('--data.train_batch_size', type=int, default=16, help='Batch size for training.')
    group.add_argument('--data.val_batch_size', type=int, default=16, help='Batch size for validation.')
    group.add_argument('--data.shuffle', type=bool, default=True, help='Whether to shuffle the data')
    group.add_argument('--drop_last', type=bool, default=True, help='Whether to drop the last incomplete batch')
    group.add_argument('--data.dataset_name', type=str, default=None, help='Dataset name')

    return parser


def transform_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for transform.
    :param parser:
    :return:
    """
    group = parser.add_argument_group(title='Arguments for transform')

    group.add_argument('--data.image_size', type=int, default=640, help='Path to dataset')

    return parser


def model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    get all arguments for models.
    :param parser:
    :return:
    """
    parser = arguments_model(parser=parser)

    return parser


def get_train_args() -> argparse.Namespace:
    parser = common_args()
    parser = cudnn_args(parser=parser)
    parser = data_args(parser=parser)
    parser = transform_args(parser=parser)
    parser = model_args(parser=parser)

    # Add training specific arguments
    parser.add_argument('--common.epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--common.learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--common.momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--common.weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--common.lr_step_size', type=int, default=30, help='Step size for learning rate scheduler')
    parser.add_argument('--common.lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')

    args = parser.parse_args()
    args = load_config_file(args)

    return args

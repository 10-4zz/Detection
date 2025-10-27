"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse

from torch.utils.data import Dataset

from utils.registry import Registry


IMAGE_FILE_TYPE = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']

DATASETS_REGISTRY = Registry(
    registry_name="DATASETS",
    component_dir=["data/datasets"],
)


def build_dataset(opts: argparse.Namespace) -> Dataset:
    """
    Build a dataset.
    """
    dataset_name = getattr(opts, "data.dataset_name", None)
    create_fn = DATASETS_REGISTRY.get(dataset_name)
    dataset = create_fn(opts)
    return dataset


def arguments_datasets(parser):
    """
    Get all arguments for datasets.
    :param parser:
    :return:
    """
    DATASETS_REGISTRY.all_arguments(parser=parser)
    return parser


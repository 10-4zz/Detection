"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from typing import Dict, Any

from torch.utils.data import Dataset

from utils.registry import Registry


DATASETS_REGISTRY = Registry(
    registry_name="DATASETS",
    component_dir=["data/datasets"],
)


def build_dataset(dataset_name: str, args: Dict[str, Any]) -> Dataset:
    """
    Build a dataset.
    """
    create_fn = DATASETS_REGISTRY.get(dataset_name)
    dataset = create_fn(**args)
    return dataset


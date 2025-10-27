"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse
from typing import Tuple

from torch.utils.data import DataLoader

from data.datasets import build_dataset


def build_dataloader(opts: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Build a dataloader.
    :param opts:
    :return:
    """
    train_dataset = build_dataset(opts, is_training=True, transform=None)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=getattr(opts, 'data.train_batch_size', 1),
        shuffle=getattr(opts, 'data.shuffle', True),
        num_workers=getattr(opts, 'data.num_workers', 0),
        drop_last=getattr(opts, 'data.drop_last', False),
    )

    val_dataset = build_dataset(opts, is_training=False, transform=None)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=getattr(opts, 'data.val_batch_size', 1),
        shuffle=False,
        num_workers=getattr(opts, 'data.num_workers', 0),
        drop_last=False,
    )

    return train_dataloader, val_dataloader

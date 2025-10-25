"""
Writen by: ian
"""
import argparse

from models.base import Base


class BaseNeck(Base):
    def __init__(
            self,
            opts: argparse.Namespace,
    ) -> None:
        super().__init__(opts)

    def forward(self, x):
        raise NotImplementedError("Please Implement forward method")

    def get_feat_index(self):
        raise NotImplementedError("Please Implement get_feat_index method")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.neck.name", type=str, default=None, help="The name of the neck.")

        return parser

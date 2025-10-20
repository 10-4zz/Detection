"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(
            self,
            path,
            img_size: int = 640,
            batch_size: int = 16,
            augment: bool = False,
            hyp=None,
            rect: bool = False,
            image_weights: bool = False,
            cache_images: bool = False,
            single_cls: bool = False,
            stride: int = 32,
            pad: float = 0.0,
            min_items: int = 0,
            prefix: str = ''
    ) -> None:
        super().__init__()


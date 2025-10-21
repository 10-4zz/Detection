"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

from utils.logger import logger


class CocoDataset(Dataset):
    """
    A standard PyTorch Dataset class for loading a COCO formatted dataset.
    """

    def __init__(self, image_dir: str, ann_file: str, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing all images (e.g., 'train2017').
            ann_file (str): Path to the COCO format annotation .json file.
            transform (callable, optional): A function/transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transform = transform

        self.coco = COCO(self.ann_file)

        self.img_ids = sorted(self.coco.getImgIds())

        self.img_ids = [img_id for img_id in self.img_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

        logger.info(f"Dataset initialized. Found {len(self.img_ids)} images with annotations in '{self.ann_file}'")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.img_ids)

    def __getitem__(self, index: int):
        """
        Retrieves a sample from the dataset at the given index.
        """
        img_id = self.img_ids[index]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x_min, y_min, w, h = ann['bbox']

            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])

            labels.append(ann['category_id'])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


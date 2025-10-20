"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import os
import hashlib
import pickle
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from utils.logger import logger


class CocoInMemory(COCO):
    def __init__(self, annotation_dict):
        super(CocoInMemory, self).__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_dict is not None:
            self.dataset = annotation_dict
            self.createIndex()


class YoloToCocoCacheDataset(Dataset):
    """
    A PyTorch Dataset that reads a YOLO-formatted dataset, converts it to COCO format
    in memory, is compatible with COCO evaluation tools, and uses a cache file
    for accelerated loading.
    """
    CACHE_VERSION = 1.0  # Change this version if you update the caching logic

    def __init__(self, image_dir, label_dir, class_names_file, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing YOLO .txt labels.
            class_names_file (str): Path to the file with class names, one per line.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names_file = class_names_file
        self.transform = transform

        self.class_names = self._load_class_names()
        self.catid2clsid = {i: i for i, name in enumerate(self.class_names)}

        cache_path = os.path.join(label_dir, f"{os.path.basename(label_dir)}.cache")

        if os.path.exists(cache_path):
            logger.info(f"Attempting to load dataset from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            current_hash = self._get_hash()
            if cached_data.get('version') == self.CACHE_VERSION and cached_data.get('hash') == current_hash:
                self.records = cached_data['records']
                coco_dict = cached_data['coco_dict']
                logger.info("Cache is valid. Loaded data successfully.")
            else:
                logger.warning("Cache is invalid or outdated. Regenerating...")
                self.records, coco_dict = self._cache_dataset(cache_path)
        else:
            logger.info("Cache not found. Creating a new one...")
            self.records, coco_dict = self._cache_dataset(cache_path)

        self.coco_api = CocoInMemory(coco_dict)
        logger.info("COCO API object for evaluation created successfully in memory.")

    def _load_class_names(self):
        with open(self.class_names_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _get_hash(self):
        files = [self.class_names_file] + sorted(os.listdir(self.label_dir))
        content = "".join(files).encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def _cache_dataset(self, cache_path):
        """
        Reads the YOLO dataset, creates data records and a COCO dict,
        and saves them to a cache file. This is the slow, one-time operation.
        """
        records = []
        coco_dict = {
            "images": [], "annotations": [],
            "categories": [{'id': i, 'name': name} for i, name in enumerate(self.class_names)],
            "info": {"description": "YOLO to COCO in-memory conversion"}, "licenses": []
        }

        ann_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        img_id_counter = 0
        ann_id_counter = 0

        for ann_file_name in tqdm(ann_files, desc="Caching dataset"):
            base_name = os.path.splitext(ann_file_name)[0]

            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']:
                potential_path = os.path.join(self.image_dir, base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if not img_path: continue

            try:
                with Image.open(img_path) as img:
                    im_w, im_h = img.size
            except Exception:
                logger.warning(f"Could not read image size for {img_path}. Skipping.")
                continue

            coco_dict['images'].append({
                'id': img_id_counter,
                'file_name': os.path.basename(img_path),
                'height': im_h,
                'width': im_w
            })

            record = {'image_path': img_path, 'height': im_h, 'width': im_w, 'labels': []}

            with open(os.path.join(self.label_dir, ann_file_name), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    class_id, x_c, y_c, w, h = [float(p) for p in parts]

                    record['labels'].append([class_id, x_c, y_c, w, h])

                    box_w = w * im_w
                    box_h = h * im_h
                    x_min = (x_c * im_w) - (box_w / 2)
                    y_min = (y_c * im_h) - (box_h / 2)
                    coco_dict['annotations'].append({
                        'id': ann_id_counter,
                        'image_id': img_id_counter,
                        'category_id': int(class_id),
                        'bbox': [x_min, y_min, box_w, box_h],
                        'area': box_w * box_h,
                        'iscrowd': 0
                    })
                    ann_id_counter += 1

            records.append(record)
            img_id_counter += 1

        # 保存到缓存文件
        cache_data = {
            'hash': self._get_hash(),
            'version': self.CACHE_VERSION,
            'records': records,
            'coco_dict': coco_dict
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"New cache created at {cache_path}")

        return records, coco_dict

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(record['image_path']).convert('RGB')

        labels = np.array(record['labels'], dtype=np.float32) if record['labels'] else np.zeros((0, 5),
                                                                                                dtype=np.float32)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    logger.info("--- Creating a dummy YOLO dataset for demonstration ---")
    os.makedirs('dummy_dataset/images/train', exist_ok=True)
    os.makedirs('dummy_dataset/labels/train', exist_ok=True)

    with open('dummy_dataset/class_names.txt', 'w') as f:
        f.write('cat\ndog\n')

    dummy_image1 = Image.new('RGB', (640, 480), color='red')
    dummy_image1.save('dummy_dataset/images/train/img1.jpg')
    with open('dummy_dataset/labels/train/img1.txt', 'w') as f:
        f.write('0 0.5 0.5 0.2 0.3\n')  # cat
        f.write('1 0.2 0.3 0.1 0.15\n')  # dog

    dummy_image2 = Image.new('RGB', (800, 600), color='blue')
    dummy_image2.save('dummy_dataset/images/train/img2.jpg')
    with open('dummy_dataset/labels/train/img2.txt', 'w') as f:
        f.write('0 0.7 0.6 0.4 0.5\n')  # cat

    logger.info("\n--- First run: Caching the dataset ---")
    dataset = YoloToCocoCacheDataset(
        image_dir='dummy_dataset/images/train',
        label_dir='dummy_dataset/labels/train',
        class_names_file='dummy_dataset/class_names.txt'
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("\n--- Second run: Loading from cache ---")
    dataset_from_cache = YoloToCocoCacheDataset(
        image_dir='dummy_dataset/images/train',
        label_dir='dummy_dataset/labels/train',
        class_names_file='dummy_dataset/class_names.txt'
    )
    logger.info(f"Dataset size: {len(dataset_from_cache)}")

    logger.info("\n--- Verifying COCO evaluation compatibility ---")
    coco_api = dataset.coco_api
    cat_ids = coco_api.getCatIds()
    cat_names = [c['name'] for c in coco_api.loadCats(cat_ids)]
    logger.info(f"Category IDs found by coco_api: {cat_ids}")
    logger.info(f"Category Names found by coco_api: {cat_names}")

    import shutil

    shutil.rmtree('dummy_dataset')
    os.remove('dummy_dataset/labels/train/train.cache')


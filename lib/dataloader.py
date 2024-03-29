
import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
               'dng', 'webp', 'mpo']  # acceptable image suffixes

class CustomDetectionDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = list(
            sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.labels = list(
            sorted(os.listdir(os.path.join(root_dir, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.images[idx])
        lbl_path = os.path.join(self.root_dir, "labels", self.labels[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = np.loadtxt(lbl_path, delimiter=" ")
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        if self.transforms is not None:
            img, boxes = self.transforms(img, boxes)
        
        boxes = torch.as_tensor(boxes[:, 1:], dtype=torch.float32)

        labels = np.loadtxt(lbl_path, delimiter=" ", usecols=(0))
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if labels.ndim < 1:
            labels = labels.unsqueeze(0)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img, target

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = list(
            sorted(os.listdir(os.path.join(root_dir, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.images[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img=img)

        return img

    def __len__(self):
        return len(self.images)

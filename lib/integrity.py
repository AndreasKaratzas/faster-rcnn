
import os
import sys

sys.path.insert(1, '../')

import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from lib.utils import collate_fn
from lib.visual import VisualTest
from lib.dataloader import CustomDataset
from lib.transformation import get_transform


def main():
    if not Path(args.dataset).is_dir():
        raise ValueError(f"Path to dataset is invalid. Value parsed {args.dataset}.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "images")):
        raise ValueError(f"Path to training image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "labels")):
        raise ValueError(f"Path to training label data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "val", "images")):
        raise ValueError(f"Path to validation image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "val", "labels")):
        raise ValueError(f"Path to validation label data does not exist.")

    # dataloader training
    train_data = CustomDataset(
        root_dir=os.path.join(args.dataset, "train"),
        transforms=get_transform(False)
    )

    # dataloader validation
    val_data = CustomDataset(
        root_dir=os.path.join(args.dataset, "val"),
        transforms=get_transform(False)
    )

    # dataloader training
    dataloader_train = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # dataloader validation
    dataloader_valid = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # data integrity check
    visualize = VisualTest()

    for images, targets in dataloader_train:
        visualize.visualize(img=images[0] * 255, boxes=targets[0].get('boxes'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrity check on object detection dataset.')
    parser.add_argument('--dataset', default='../data/balloon', type=str, help='Path to dataset.')
    args = parser.parse_args()

    main()

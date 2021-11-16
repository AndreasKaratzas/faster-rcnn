
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import ThreadPool
from typing import List, Dict
from pathlib import Path
from PIL import Image

NUM_THREADS = 8

def find_img_files(path: Path):
    return List[str]

def get_lbl_files(img_files: List[str]):
    return List[str]

def set_cache_file(path: Path):
    return Path(path).with_suffix('.cache')

def load_cache(path: Path):
    return np.load(path, allow_pickle=True).item()

def lbl2img_path(lbl_path: Path, img_path: Path):
    return lbl_path.stem == img_path.stem

def get_aux_inf(path: Path):
    return np.ndarray, np.ndarray, int, float, np.ndarray

def get_hash(images: List[str], labels: List[str]):
    hash = str
    return hash

def get_cache_stats(images: List[str], labels: List[str]):
    num_found, num_missing, num_empty, num_corrupt, num_total = int, int, int, int, int
    return num_found, num_missing, num_empty, num_corrupt, num_total

def cache_labels(path: Path, images: List[str], labels: List[str]):
    X = {}
    
    for img_sample, lbl_sample in zip(images, labels):
        X[img_sample] = {}

        boxes, classes, image_id, area, iscrowd = get_aux_inf(lbl_sample)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        X[img_sample] = target
    
    X["hash"] = get_hash(images, labels)

    np.save(path, X)
    path.with_suffix('.cache.npy').rename(path)

def read_cache(cache):
    cache.pop("hash")
    labels = list(zip(*cache.values()))
    img_files = list(cache.keys())
    label_files = get_lbl_files(cache.keys())

def load_image(img_idx: int, img_dataset: List[np.ndarray], img_cache: List[np.ndarray], img_filelist: List[str]):
    # check if image is already cached
    if img_dataset[img_idx] is None:
        # check if image is already loaded
        npy = img_cache[img_idx]

        if npy and npy.exists():
            # cache loaded image
            img = np.load(npy)
        else:
            # load image and cache it
            img = cv2.imread(img_filelist[img_idx])
        
        return img
    else:
        return img_dataset[img_idx]

def cache_images(num_found: int):
    gb, imgs = 0, [None] * num_found
    
    results = ThreadPool(NUM_THREADS)
    results.map(lambda x: load_image(*x), range(num_found))

    for sample_idx, sample_img in enumerate(results):
        imgs[sample_idx] = sample_img
        gb += imgs[sample_idx].nbytes

def get_sample(sample_idx: int, targets: Dict[Dict]):
    # load data
    img = load_image(sample_idx)
    targets = targets[sample_idx].copy()
    # augment data
    # return data

def build_dataloader(dataset):
    pass
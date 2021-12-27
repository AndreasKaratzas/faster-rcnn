
import os
import re
import glob
import psutil
import torch
import hashlib
from multiprocessing.pool import Pool, ThreadPool
import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import deque
from itertools import repeat
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from lib.presets import DetectionPresetTargetOnlyTorchVision


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
               'dng', 'webp', 'mpo']


def _verify_lbl2img_path(args):
    img_file, lbl_file = args
    msg = ' '

    try:
        # load image
        img = Image.open(img_file)
        # verify integrity
        img.verify()
        # register image dimensions
        width, height = img.size
        # check file extension
        assert img.format.lower(
        ) in IMG_FORMATS, f'invalid image format {img.format}'
        # try to restore corrupt jpeg
        if img.format.lower() in ('jpg', 'jpeg'):
            with open(img_file, 'rb') as f:
                f.seek(-2, 2)
                # if image is corrupt
                if f.read() != b'\xff\xd9':
                    # restore image file
                    ImageOps.exif_transpose(Image.open(img_file)).save(
                        img_file, 'JPEG', subsampling=0, quality=100)
                    msg += f'WARNING: image with ID {str(Path(img_file).stem)}: corrupt JPEG restored and saved.\n'
        # verify labels
        if os.path.isfile(lbl_file):
            # load labels
            lbl = np.loadtxt(lbl_file, delimiter=" ")
            # convert to torch tensor
            lbl = torch.as_tensor(lbl, dtype=torch.float32)
            # unsqueeze labels in case of single object
            if lbl.ndim == 1:
                lbl = lbl.unsqueeze(0)
            # get number of labels
            num_lbls = len(lbl)
            # check for empty labels file
            if lbl is not None:
                # check label file structure
                assert lbl.shape[1] == 5, f'Labels require 5 columns, {lbl.shape[1]} columns detected'
                # check for duplicate labels
                _, dupl_idx_ndarray = np.unique(lbl, axis=0, return_index=True)
                # if duplicate labels were found
                if len(dupl_idx_ndarray) < num_lbls:
                    # filter out duplicates
                    lbl = lbl[dupl_idx_ndarray]
                    msg += f'WARNING: image with ID {str(Path(img_file).stem)}: {num_lbls - len(dupl_idx_ndarray)} duplicate label(s) removed.\n'
            else:
                raise ValueError(
                    f"Found empty label file. ID {Path(lbl_file).stem}.")
        # label file was not found
        else:
            raise ValueError(
                f"Missing label file for image with ID {str(Path(img_file).stem)}.")
        return img_file, lbl, width, height, msg
    except Exception as e:
        raise AttributeError(
            f"Image and/or label file is corrupt regarding sample with ID {Path(img_file).stem}.")


def _load_image(self, img_idx: int, caching = True):
    """Processes the `images` attribute of `CustomDetectionDataset` object
    Parameters
    ----------
    img_idx : int
        [description]
    """
    if self.images[img_idx] is not None:
        # if img exists in cache
        return self.images[img_idx]
    else:
        if not caching:
            cntr = 0
            min_idx = 1000000
            max_idx = 0
            for idx, img in enumerate(self.images):
                if img is not None:
                    cntr += 1
                    if min_idx > idx:
                        min_idx = idx
                    if max_idx < idx:
                        max_idx = idx
            print(f"Cached from {min_idx} to {max_idx}.")
            print(f"{cntr} images cached.")
            print(f"{img_idx} NOT CACHED !!!")
        # fetch if it does not exist
        img_path = self.img_files[img_idx]
        # load image sample
        img = Image.open(img_path).convert("RGB")
        # reduce image dimensions
        img = img.resize((self.img_size, self.img_size), Image.NEAREST)
        # assert error if image was not found
        assert img is not None, f'Image Not Found {img_path}'
        # return result
        return img


class CustomCachedDetectionDataset(Dataset):
    def __init__(self, root_dir, num_threads: int = 8, batch_size: int = 16, img_size: int = 640, transforms=None, cache_images_flag: bool = True):
        # Solves "OSError: Too many open files."
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.root_dir = root_dir
        self.transforms = transforms
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.img_size = img_size
        self.cache_images_flag = cache_images_flag
        self.reduce_target = DetectionPresetTargetOnlyTorchVision(
            img_size=self.img_size)

        self._fetch_img_files(root_dir=root_dir)
        self._fetch_lbl_files(img_files=self.img_files)

        cache_path = Path(root_dir).with_suffix('.cache')
        self._config_cache(cache_path=cache_path)

    def _fetch_img_files(self, root_dir: str):
        f = glob.glob(str(Path(root_dir) / '**' / '*.*'), recursive=True)
        self.img_files = sorted(x.replace('/', os.sep)
                                for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

    def _fetch_lbl_files(self, img_files: List[Path]):
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        self.lbl_files = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[
            0] + '.txt' for x in img_files]

    def _get_hash(self):
        paths = self.img_files + self.lbl_files
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        h = hashlib.md5(str(size).encode())
        h.update(''.join(paths).encode())
        return h.hexdigest()

    def _compute_image_placeholders(self):
        image_idx = 0
        # fetch if it does not exist
        img_path = self.img_files[image_idx]
        # load image sample
        img = Image.open(img_path).convert("RGB")
        # reduce image dimensions
        img = img.resize((self.img_size, self.img_size), Image.NEAREST)
        # assert error if image was not found
        assert img is not None, f'Image Not Found {image_idx}'
        # get ram requirements for a single sample
        _allocated_mem = np.asarray(img).nbytes
        # get total ram memory
        _available_ram_space = psutil.virtual_memory().available * 0.8
        # compute expected ram requirements
        _expected_ram_reqs = np.ceil(
            self.num_samples * _allocated_mem * 1.1)
        # estimate number of image placeholders
        self.num_of_image_placeholders = np.ceil(
            _expected_ram_reqs / _available_ram_space).astype(int)
        # reconfigure number of image placeholders to handle with threading
        self.num_of_image_placeholders = self.num_of_image_placeholders * 2 \
            if self.num_of_image_placeholders > 1 else 1
        if self.num_of_image_placeholders > 1:
            # set number of image samples per image placeholder
            _num_of_img_per_placeholder = np.ceil(
                self.num_samples / self.num_of_image_placeholders).astype(int)
            # precompute the exact number of image samples per image placeholder
            self.img_per_placeholder_lst = [
                _num_of_img_per_placeholder] * (self.num_of_image_placeholders - 1)
            # configure first index
            self.img_per_placeholder_lst.insert(0, 0)
            # configure last index
            self.img_per_placeholder_lst.append(self.num_samples -
                                                (_num_of_img_per_placeholder * (self.num_of_image_placeholders - 1)))
            # index segment covered by each image placeholder
            self.img_idx_segment_per_placeholer = [np.sum(
                self.img_per_placeholder_lst[:idx]) for idx in range(1, len(self.img_per_placeholder_lst) + 1)]
            # segments with indexes corresponding to every image placeholder
            self.segments_with_indexes_per_img_placeholder = [list(np.arange(
                self.img_idx_segment_per_placeholer[idx - 1], self.img_idx_segment_per_placeholer[idx])) 
                for idx in range(1, len(self.img_idx_segment_per_placeholer))
            ]
            # initialize a container with subset indexes to be cached
            self.subset_to_be_cached_idx = deque(
                range(len(self.segments_with_indexes_per_img_placeholder)))

    def _cache_labels(self, cache_path: Path):
        x, msgs = {}, []

        desc = f"Scanning '{Path(cache_path.parent.name) / Path(cache_path.stem)}' directory for images and labels"

        if self.num_threads > 1:
            with ThreadPool(self.num_threads) as pool:
                pbar = tqdm(pool.imap(_verify_lbl2img_path, zip(
                    self.img_files, self.lbl_files)), desc=desc, total=len(self.img_files), unit=" samples processed")
                # verify target files w.r.t. images found in the dataset
                for img_file, lbl, width, height, msg in pbar:
                    # reduce target dimensions w.r.t. image dimensionality reduction ratio
                    lbl = self.reduce_target(
                        target=lbl, height=height, width=width)
                    # avoid invalid boxes
                    lbl[:, 3] += 1
                    lbl[:, 4] += 1
                    # record targets or message indicating a failure in the parsing process
                    if img_file:
                        x[img_file] = lbl
                    if msg:
                        msgs.append(msg)
            pbar.close()
        else:
            pbar = tqdm(zip(self.img_files, self.lbl_files), desc=desc,
                        total=len(self.img_files), unit=" samples processed")
            for img_file, lbl_file in pbar:
                # verify target files w.r.t. images found in the dataset
                img_file, lbl, width, height, msg = _verify_lbl2img_path(
                    (img_file, lbl_file))
                # reduce target dimensions w.r.t. image dimensionality reduction ratio
                lbl = self.reduce_target(
                    target=lbl, height=height, width=width)
                # avoid invalid boxes
                lbl[:, 3] += 1
                lbl[:, 4] += 1
                # record targets or message indicating a failure in the parsing process
                if img_file:
                    x[img_file] = lbl
                if msg:
                    msgs.append(msg)
            pbar.close()

        return x, msgs

    def _cache_images(self, verbose: bool = False):
        # register memory allocated in RAM
        _allocated_mem = 0
        # delete second from last subset from cache
        self.images[self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[-2]][0]:
                    self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[-2]][-1]] = [None] * (
                    self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[-2]][-1] -
                    self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[-2]][0])
        # cache in a subset
        if self.num_threads > 1:
            # initialize multithreaded image fetching operation
            _results = ThreadPool(self.num_threads).imap(
                lambda x: _load_image(*x), zip(
                    repeat(
                        self, 
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] -
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0] + 1
                    ), 
                    range(
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0],
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] + 1)
                )
            )
            if verbose:
                # keep user informed with a TQDM bar
                pbar = tqdm(enumerate(_results), 
                            total=self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] -
                            self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0] + 1,
                            unit=" samples processed")
                # loop through samples
                for image_idx, image_sample in pbar:
                    # cache image
                    self.images[self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]]
                                [image_idx]] = image_sample
                    # update allocated memory register
                    _allocated_mem += np.asarray(
                        self.images[self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][image_idx]]).nbytes
                    # update RAM status
                    pbar.desc = f"Caching images ({(self._init_allocated_mem + _allocated_mem) / 1E9: .3f} GB RAM)"
                pbar.close()
            else:
                # loop through samples
                for image_idx, image_sample in enumerate(_results):
                    # cache image
                    self.images[self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]]
                                            [image_idx]] = image_sample
        else:
            if verbose:
                # keep user informed with a TQDM bar
                pbar = tqdm(range(
                            self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0],
                            self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] + 1),
                            total=self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] -
                            self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0] + 1,
                            unit=" samples processed")
                # initialize single threaded image fetching operation
                for image_idx in pbar:
                    # fetch if it does not exist
                    img_path = self.img_files[image_idx]
                    # load image sample
                    img = Image.open(img_path).convert("RGB")
                    # reduce image dimensions
                    img = img.resize((self.img_size, self.img_size), Image.NEAREST)
                    # assert error if image was not found
                    assert img is not None, f'Image Not Found {image_idx}'
                    # cache image
                    self.images[image_idx] = img
                    # update allocated memory register
                    _allocated_mem += np.asarray(
                        self.images[image_idx]).nbytes
                    # update RAM status
                    pbar.desc = f"Caching images({(self._init_allocated_mem + _allocated_mem) / 1E9: .3f} GB RAM)"
                pbar.close()
            else:
                # initialize single threaded image fetching operation
                for image_idx in range(
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][0],
                        self.segments_with_indexes_per_img_placeholder[self.subset_to_be_cached_idx[0]][-1] + 1):
                    # fetch if it does not exist
                    img_path = self.img_files[image_idx]
                    # load image sample
                    img = Image.open(img_path).convert("RGB")
                    # reduce image dimensions
                    img = img.resize(
                        (self.img_size, self.img_size), Image.NEAREST)
                    # assert error if image was not found
                    assert img is not None, f'Image Not Found {image_idx}'
                    # cache image
                    self.images[image_idx] = img
        # update previously allocated space register
        self._init_allocated_mem = _allocated_mem
        # rotate negatively the container with subset indexes to be cached
        self.subset_to_be_cached_idx.rotate(-1)

    def _config_cache(self, cache_path: Path):
        # create cache
        cache, msgs = self._cache_labels(cache_path)

        # print warnings
        if msgs:
            msgs = "".join(msgs)
            msgs = re.sub(' +', ' ', msgs)
            msgs = re.sub('\n+', '\n', msgs)
            print(f"{msgs}")

        # extract labels (List[np.ndarray])
        self.labels = list(cache.values())

        # update images' paths
        self.img_files = list(cache.keys())
        # update labels' paths
        self.lbl_files = self._fetch_lbl_files(cache.keys())

        # configure hyperparameters
        self.num_samples = len(self.img_files)
        self.batch_index = np.floor(
            np.arange(self.num_samples) / self.batch_size).astype(np.int64)
        self.num_batches = self.batch_index[-1] + 1
        self.sample_idxs = range(self.num_samples)

        # declare cache variable for images
        self.images = [None] * self.num_samples

        self._compute_image_placeholders()
        # register previously allocated memory space
        self._init_allocated_mem = 0
        # cache images
        if self.cache_images_flag:
            # cache first subset
            self._cache_images(verbose=True)
            # cache second subset
            self._cache_images(verbose=True)

    def __getitem__(self, idx):
        img = _load_image(self, idx, caching=False)

        boxes = self.labels[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float64)

        labels = torch.as_tensor(boxes[:, 0], dtype=torch.int64)
        boxes = torch.as_tensor(boxes[:, 1:], dtype=torch.float64)

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.num_samples

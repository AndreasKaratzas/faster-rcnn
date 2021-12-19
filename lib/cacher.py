
import os
import glob
import torch
import hashlib
import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from itertools import repeat
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from multiprocessing.pool import Pool, ThreadPool

from lib.presets import DetectionPresetTargetOnlyTorchVision


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
               'dng', 'webp', 'mpo']

def _verify_lbl2img_path(args):
    img_file, lbl_file = args
    msg = ''

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
                    msg += f'WARNING: {img_file}: corrupt JPEG restored and saved\n'
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
                    msg += f'WARNING: {img_file}: {num_lbls - len(dupl_idx_ndarray)} duplicate labels removed.\n'
            else:
                raise ValueError(
                    f"Found empty label file. ID {Path(lbl_file).stem}.")
        # label file was not found
        else:
            raise ValueError(f"Missing label file for image {img_file}.")
        return img_file, lbl, width, height, msg
    except Exception as e:
        raise AttributeError(
            f"Image and/or label file is corrupt regarding sample with ID {Path(img_file).stem}.")


def _load_image(self, img_idx: int, cached: bool = False):
    """Processes the `images` attribute of `CustomDetectionDataset` object

    Parameters
    ----------
    img_idx : int
        [description]
    """
    if cached:
        img = self.images[img_idx]
        # check if img exists in cache
        return self.images[img_idx]
    else:
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
        self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

    def _fetch_lbl_files(self, img_files: List[Path]):
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        self.lbl_files = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_files]

    def _get_hash(self):
        paths = self.img_files + self.lbl_files
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        h = hashlib.md5(str(size).encode())
        h.update(''.join(paths).encode())
        return h.hexdigest()

    def _cache_labels(self, cache_path: Path):
        x, msgs = {}, []

        desc = f"Scanning '{cache_path.parent.name}' directory for images and labels"

        if self.num_threads > 1:
            with Pool(self.num_threads) as pool:
                pbar = tqdm(pool.map(_verify_lbl2img_path, zip(
                    self.img_files, self.lbl_files)), desc=desc, total=len(self.img_files), unit=" samples processed")
                # verify target files w.r.t. images found in the dataset
                for img_file, lbl, width, height, msg in pbar:
                    # reduce target dimensions w.r.t. image dimensionality reduction ratio
                    lbl = self.reduce_target(
                        target=lbl, height=height, width=width)
                    # avoid invalid boxes
                    lbl[2] += 1
                    lbl[3] += 1
                    # record targets or message indicating a failure in the parsing process
                    if img_file:
                        x[img_file] = lbl
                    if msg:
                        msgs.append(msg)
            pbar.close()
        else:
            pbar = tqdm(zip(self.img_files, self.lbl_files), desc=desc, total=len(self.img_files), unit=" samples processed")
            for img_file, lbl_file in pbar:
                # verify target files w.r.t. images found in the dataset
                img_file, lbl, width, height, msg = _verify_lbl2img_path((img_file, lbl_file))
                # reduce target dimensions w.r.t. image dimensionality reduction ratio
                lbl = self.reduce_target(
                    target=lbl, height=height, width=width)
                # avoid invalid boxes
                lbl[2] += 1
                lbl[3] += 1
                # record targets or message indicating a failure in the parsing process
                if img_file:
                    x[img_file] = lbl
                if msg:
                    msgs.append(msg)
            pbar.close()

        # print warnings
        if msgs:
            print(f"{' '.join(msgs)}")

        return x

    def _cache_images(self):
        # declare cache variable for images
        self.images = [None] * self.num_samples
        # register memory allocated in RAM
        _allocated_mem = 0
        if self.num_threads > 1:
            # initialize multithreaded image fetching operation
            _results = ThreadPool(self.num_threads).imap(
                lambda x: _load_image(*x), zip(repeat(self, self.num_samples), range(self.num_samples)))
            # keep user informed with a TQDM bar
            pbar = tqdm(enumerate(_results), total=self.num_samples, unit=" samples processed")
            # loop through samples
            for image_idx, image_sample in pbar:
                # cache image
                self.images[image_idx] = image_sample
                # update allocated memory register
                _allocated_mem += np.asarray(self.images[image_idx]).nbytes
                # update RAM status
                pbar.desc = f"Caching images ({_allocated_mem / 1E9: .3f}GB RAM)"
            pbar.close()
        else:
            # keep user informed with a TQDM bar
            pbar = tqdm(range(self.num_samples), total=self.num_samples,
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
                _allocated_mem += np.asarray(self.images[image_idx]).nbytes
                # update RAM status
                pbar.desc = f"Caching images({_allocated_mem / 1E9: .3f}GB RAM)"
            pbar.close()


    def _config_cache(self, cache_path: Path):
        # create cache
        cache = self._cache_labels(cache_path)
        
        # extract labels (List[np.ndarray])
        self.labels = list(cache.values())

        # update images' paths
        self.img_files = list(cache.keys())
        # update labels' paths
        self.lbl_files = self._fetch_lbl_files(cache.keys())

        # configure hyperparameters
        self.num_samples = len(self.img_files)
        self.batch_index = np.floor(np.arange(self.num_samples) / self.batch_size).astype(np.int64)
        self.num_batches = self.batch_index[-1] + 1
        self.sample_idxs = range(self.num_samples)

        # cache images
        if self.cache_images_flag:
            self._cache_images()

    def __getitem__(self, idx):
        img = _load_image(self, idx, cached=self.cache_images_flag)
        
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

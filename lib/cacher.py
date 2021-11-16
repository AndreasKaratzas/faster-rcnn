
import os
import cv2
import glob
import torch
import hashlib
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from itertools import repeat
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from multiprocessing.pool import Pool, ThreadPool

from lib.presets import DetectionPresetImageOnly, DetectionPresetTargetOnly


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
            num_lbls = len(lbl)
            # check for empty labels file
            if not lbl:
                raise ValueError(
                    f"Found empty label file. ID {lbl_file.stem}.")
            else:
                # check label file structure
                assert lbl.shape[1] == 5, f'Labels require 5 columns, {lbl.shape[1]} columns detected'
                # check for duplicate labels
                _, dupl_idx_ndarray = np.unique(lbl, axis=0, return_index=True)
                # if duplicate labels were found
                if len(dupl_idx_ndarray) < num_lbls:
                    # filter out duplicates
                    lbl = lbl[dupl_idx_ndarray]
                    msg += f'WARNING: {img_file}: {num_lbls - len(dupl_idx_ndarray)} duplicate labels removed.\n'
        # label file was not found
        else:
            raise ValueError(f"Missing label file for image {img_file}.")
        return img_file, lbl, msg
    except Exception as e:
        raise AttributeError(f"Image and/or label file is corrupt regarding sample with ID {img_file.stem}.")


def _load_image(self, img_idx: int):
    """Processes the `images` attribute of `CustomDetectionDataset` object

    Parameters
    ----------
    img_idx : int
        [description]
    """
    img = self.images[img_idx]
    # check if img exists in cache
    if img is None:
        # fetch if it does not exist
        img_path = self.img_files[img_idx]
        # load image sample
        img = cv2.imread(img_path)
        # transpose BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # reduce image dimensions
        img = self.reduce_image(image=img)
        # assert error if image was not found
        assert img is not None, f'Image Not Found {img_path}'
        # return result
        return img
    else:
        return self.images[img_idx]

class CustomDetectionDataset(Dataset):
    def __init__(self, root_dir, num_threads: int = 8, batch_size: int = 16, img_size: int = 640, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.img_size = img_size
        self.reduce_image = DetectionPresetImageOnly(img_size=self.img_size)
        self.reduce_target = DetectionPresetTargetOnly(img_size=self.img_size)
        
        self._fetch_img_files(root_dir=root_dir)
        self._fetch_lbl_files()

        cache_path = Path(root_dir).with_suffix('.cache')
        self._config_cache(cache_path=cache_path)
        
    def _fetch_img_files(self, root_dir: str):
        f = glob.glob(str(Path(root_dir) / '**' / '*.*'), recursive=True)
        self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

    def _fetch_lbl_files(self):
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        self.lbl_files = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in self.img_files]

    def _get_hash(self):
        paths = self.img_files + self.lbl_files
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        h = hashlib.md5(str(size).encode())
        h.update(''.join(paths).encode())
        return h.hexdigest()

    def _cache_labels(self, cache_path: Path):
        x, msgs = {}, []

        desc = f"Scanning '{cache_path.parent.name}' directory for images and labels"
        with Pool(self.num_threads) as pool:
            pbar = tqdm(pool.imap(_verify_lbl2img_path, zip(
                self.img_files, self.lbl_files)), desc=desc, total=len(self.img_files))
            for img_file, lbl, msg in pbar:
                # reduce target dimensions w.r.t. image dimensionality reduction ratio
                lbl = self.reduce_target(bboxes=lbl)
                if img_file:
                    x[img_file] = lbl
                if msg:
                    msgs.append(msg)
        pbar.close()

        # print warnings
        if msgs:
            pprint(msgs)
        
        # include hash key
        x["hash"] = self._get_hash()
        # include messages
        x["msgs"] = msgs

        # store cache
        try:
            np.save(cache_path, x)
            cache_path.with_suffix('.cache.npy').rename(cache_path)
            print(f"Created new cache at path: {cache_path}")
            return x
        except Exception as e:
            raise ValueError(f"Cache directory not writeable: {e}")

    def _cache_images(self):
        # declare cache variable for images
        self.images = [None] * self.num_samples
        # register memory allocated in RAM
        _allocated_mem = 0
        # initialize multithreaded image fetching operation
        _results = ThreadPool(self.num_threads).imap(
            lambda x: _load_image(*x), zip(repeat(self), range(self.num_samples)))
        # keep user informed with a TQDM bar
        pbar = tqdm(enumerate(_results), total=self.num_samples)
        # loop through samples
        for image_idx, image_sample in pbar:
            # cache image
            self.images[image_idx] = image_sample
            # update allocated memory register
            _allocated_mem += self.images[image_idx].nbytes
            # update RAM status
            pbar.desc = f"Caching images({_allocated_mem / 1E9: .3f}GB RAM)"
        pbar.close()

    def _config_cache(self, cache_path: Path):
        # cache exists
        try:
            # load cache
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache["hash"] == self._get_hash()
        # cache does not exist
        except:
            # create cache
            cache, exists = self._cache_labels(cache_path), False
        
        # check past version cache integrity
        if exists:
            if cache["msgs"]:
                pprint(cache["msgs"])
        
        # clear cache from unnecessary attributes
        [cache.pop(k) for k in ("hash", "msgs")]

        # extract labels (np.ndarray)
        self.labels = zip(*cache.values())

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
        self._cache_images()

    def __getitem__(self, idx):
        img = _load_image(self, idx)
        
        boxes = self.labels[idx].copy()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        if self.transforms is not None:
            img, boxes = self.transforms(img, boxes)

        labels = torch.as_tensor(boxes[:, 0], dtype=torch.int64)
        boxes = torch.as_tensor(boxes[:, 1:], dtype=torch.float32)

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
    
    @staticmethod
    def collate_fn_embedded(batch):
        img, label = zip(*batch)
        return torch.stack(img, 0), torch.cat(label, 0)

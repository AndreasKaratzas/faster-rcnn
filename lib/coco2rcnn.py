
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import wget
from tqdm import tqdm


class installCocoDataset:
    def __init__(self, urls: List[str], res_dir: str):
        self.urls = urls
        self.res_dir = res_dir
        self.init_flag = True
        self.filepaths = []

    def configure(self):
        Path(self.res_dir).mkdir(parents=True, exist_ok=True)

        train_images_path = Path(self.res_dir) / Path("train") / Path("images")
        train_labels_path = Path(self.res_dir) / Path("train") / Path("labels")
        valid_images_path = Path(self.res_dir) / Path("valid") / Path("images")
        valid_labels_path = Path(self.res_dir) / Path("valid") / Path("labels")

        Path(train_images_path).mkdir(parents=True, exist_ok=True)
        Path(train_labels_path).mkdir(parents=True, exist_ok=True)
        Path(valid_images_path).mkdir(parents=True, exist_ok=True)
        Path(valid_labels_path).mkdir(parents=True, exist_ok=True)

        for url in self.urls:
            self.filepaths.append(
                str(Path(self.res_dir) / Path(url).parts[-1]))

    def download(self):
        for idx, url in enumerate(self.urls):
            wget.download(url, out=self.filepaths[idx], bar=wget.bar_adaptive)

    def extract(self):
        for filepath in self.filepaths:
            with zipfile.ZipFile(filepath) as zf:
                for member in tqdm(zf.infolist(), desc="Extracting", unit=" files/sec"):
                    try:
                        zf.extract(member, str(
                            Path(self.res_dir) / Path(filepath).stem))
                    except zipfile.error as e:
                        pass

    def convert(self, sample):
        x1, y1, w, h = sample['bbox']
        return x1, y1, x1 + w, y1 + h

    def clean(self):
        filenames = os.listdir(os.path.join(
            self.res_dir, "train2017", "train2017"))

        for filename in tqdm(
            iterable=filenames,
            desc="Moving training sample images",
            unit=" samples/sec"
        ):
            shutil.move(os.path.join(self.res_dir, "train2017", "train2017", filename), os.path.join(
                self.res_dir, "train", "images", filename))

        filenames = os.listdir(os.path.join(
            self.res_dir, "val2017", "val2017"))

        for filename in tqdm(
            iterable=filenames,
            desc="Moving validation sample images",
            unit=" samples/sec"
        ):
            shutil.move(os.path.join(self.res_dir, "val2017", "val2017", filename), os.path.join(
                self.res_dir, "valid", "images", filename))

        shutil.rmtree(os.path.join(self.res_dir, "annotations_trainval2017"))
        shutil.rmtree(os.path.join(self.res_dir, "train2017"))
        shutil.rmtree(os.path.join(self.res_dir, "val2017"))

        os.remove(os.path.join(self.res_dir, "annotations_trainval2017.zip"))
        os.remove(os.path.join(self.res_dir, "train2017.zip"))
        os.remove(os.path.join(self.res_dir, "val2017.zip"))

    def preprocess(self):
        filenames = os.listdir(os.path.join(
            self.res_dir, "train", "images"))

        for filename in tqdm(
            iterable=filenames,
            desc="Removing empty training samples",
            unit=" samples/sec"
        ):
            if os.stat(os.path.join(self.res_dir, "train", "images", filename)).st_size == 0:
                os.remove(os.path.join(self.res_dir,
                          "train", "images", filename))

        filenames = os.listdir(os.path.join(
            self.res_dir, "valid", "images"))

        for filename in tqdm(
            iterable=filenames,
            desc="Removing empty validation samples",
            unit=" samples/sec"
        ):
            if os.stat(os.path.join(self.res_dir, "valid", "images", filename)).st_size == 0:
                os.remove(os.path.join(self.res_dir,
                          "valid", "images", filename))

        train_image_dir = Path(self.res_dir) / "train" / "images"
        train_label_dir = Path(self.res_dir) / "train" / "labels"
        valid_image_dir = Path(self.res_dir) / "valid" / "images"
        valid_label_dir = Path(self.res_dir) / "valid" / "labels"

        train_images = Path(train_image_dir).glob('*.*')
        train_labels = Path(train_label_dir).glob('*.txt')
        valid_images = Path(valid_image_dir).glob('*.*')
        valid_labels = Path(valid_label_dir).glob('*.txt')

        train_img_filenames = [f.stem for f in train_images]
        train_lbl_filenames = [f.stem for f in train_labels]
        valid_img_filenames = [f.stem for f in valid_images]
        valid_lbl_filenames = [f.stem for f in valid_labels]

        train_img_sup = list(set(train_img_filenames) -
                             set(train_lbl_filenames))
        valid_img_sup = list(set(valid_img_filenames) -
                             set(valid_lbl_filenames))
        train_lbl_sup = list(set(train_lbl_filenames) -
                             set(train_img_filenames))
        valid_lbl_sup = list(set(valid_lbl_filenames) -
                             set(valid_img_filenames))

        if train_img_sup:
            train_images = os.listdir(train_image_dir)

            for train_img in tqdm(
                iterable=train_img_sup,
                desc="Removing redundant training image samples",
                unit=" samples/sec"
            ):
                res = list(filter(lambda x: train_img in x, train_images))
                os.remove(os.path.join(self.res_dir,
                                       "train", "images", res[0]))

        if valid_img_sup:
            valid_images = os.listdir(valid_image_dir)

            for valid_img in tqdm(
                iterable=valid_img_sup,
                desc="Removing redundant validation image samples",
                unit=" samples/sec"
            ):
                res = list(filter(lambda x: valid_img in x, valid_images))
                os.remove(os.path.join(self.res_dir,
                                       "valid", "images", res[0]))

        if train_lbl_sup:
            train_labels = os.listdir(train_label_dir)
            train_labels = [
                fname for fname in train_labels if fname.endswith('.txt')]

            for train_lbl in tqdm(
                iterable=train_lbl_sup,
                desc="Removing redundant training label samples",
                unit=" samples/sec"
            ):
                res = list(filter(lambda x: train_lbl in x, train_labels))
                os.remove(os.path.join(self.res_dir,
                                       "train", "labels", res[0]))

        if valid_lbl_sup:
            valid_labels = os.listdir(valid_label_dir)
            valid_labels = [
                fname for fname in valid_labels if fname.endswith('.txt')]

            for valid_lbl in tqdm(
                iterable=valid_lbl_sup,
                desc="Removing redundant validation label samples",
                unit=" samples/sec"
            ):
                res = list(filter(lambda x: valid_lbl in x, valid_labels))
                os.remove(os.path.join(self.res_dir,
                                       "valid", "labels", res[0]))

    def install(self, annotation_filepath: str, out_filepath):

        dataset_filepath = open(annotation_filepath)
        dataset_data = json.load(dataset_filepath)

        annotations = []

        for annotation in tqdm(
            iterable=dataset_data['annotations'],
            desc="Converting COCO dataset to Faster R-CNN format",
            unit=" labels/sec"
        ):
            image_id = annotation['image_id']

            if self.init_flag:
                self.prev_image_id = image_id
                self.init_flag = False

            x1, y1, x2, y2 = self.convert(annotation)
            annotation_bbox = np.ceil(np.array([x1, y1, x2, y2]))
            annotation_id = np.array([annotation['category_id']])
            annotation_sample = np.append(
                annotation_id.astype(str), annotation_bbox.astype(str))

            if self.prev_image_id == image_id:
                annotations.append(annotation_sample)
            else:
                np.savetxt(os.path.join(
                    out_filepath, str(self.prev_image_id).zfill(12) + '.txt'), annotations, fmt='%s')
                annotations = []
                annotations.append(annotation_sample)
                self.prev_image_id = image_id

        dataset_filepath.close()

    def driver(self):
        self.configure()
        self.download()
        self.extract()

        self.install(
            annotation_filepath=os.path.join(
                self.res_dir,
                'annotations_trainval2017',
                'annotations',
                'instances_train2017.json'
            ),
            out_filepath=os.path.join(
                self.res_dir,
                'train',
                'labels'
            )
        )

        self.install(
            annotation_filepath=os.path.join(
                self.res_dir,
                'annotations_trainval2017',
                'annotations',
                'instances_val2017.json'
            ),
            out_filepath=os.path.join(
                self.res_dir,
                'valid',
                'labels'
            )
        )

        self.clean()
        self.preprocess()


url_train_imgs = "http://images.cocodataset.org/zips/train2017.zip"
url_valid_imgs = "http://images.cocodataset.org/zips/val2017.zip"
url_lbls = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

res_dir = Path("../datasets/coco")

coco_installer = installCocoDataset(
    urls=[url_train_imgs, url_valid_imgs, url_lbls], res_dir=res_dir)
coco_installer.driver()

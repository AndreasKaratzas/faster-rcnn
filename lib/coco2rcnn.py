
import os
import shutil
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import wget
from PIL import Image
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
                for member in tqdm(zf.infolist(), desc="Extracting", unit=" files"):
                    try:
                        zf.extract(member, str(
                            Path(self.res_dir) / Path(filepath).stem))
                    except zipfile.error as e:
                        pass

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

        shutil.rmtree(os.path.join(self.res_dir, "train2017"))
        shutil.rmtree(os.path.join(self.res_dir, "val2017"))

        os.remove(os.path.join(self.res_dir, "train2017.zip"))
        os.remove(os.path.join(self.res_dir, "val2017.zip"))

    def rotate(self, image):
        rot_flag = False
        exif = image.getexif()
        orientation = exif.get(0x0112, 1)  # default 1
        if orientation > 1:
            method = {
                2: Image.FLIP_LEFT_RIGHT,
                3: Image.ROTATE_180,
                4: Image.FLIP_TOP_BOTTOM,
                5: Image.TRANSPOSE,
                6: Image.ROTATE_270,
                7: Image.TRANSVERSE,
                8: Image.ROTATE_90,
            }.get(orientation)
            if method is not None:
                rot_flag = True
                image = image.transpose(method)
                del exif[0x0112]
                image.info["exif"] = exif.tobytes()
        return rot_flag, image

    def preprocess(self):

        subsets = ['train', 'valid']

        for subset_id in subsets:
            rot_cntr = 0
            file_not_found_cntr = 0
            imgs_lst = os.listdir(os.path.join(
                self.res_dir, subset_id, "images"))
            for img_path in tqdm(
                iterable=imgs_lst,
                desc=f"Parsing {subset_id} image subfolders",
                total=len(imgs_lst),
                unit=" samples"
            ):

                try:
                    label = np.loadtxt(
                        os.path.join(
                            self.res_dir, 
                            subset_id, "labels", 
                            str(Path(img_path).resolve().stem) + ".txt"
                        )
                    )
                    
                    img = Image.open(os.path.join(
                        self.res_dir, subset_id, "images", img_path))
                    rotation_flag, img = self.rotate(img)

                    if rotation_flag:
                        rot_cntr += 1
                        os.remove(os.path.join(
                            self.res_dir, subset_id, "images", img_path))
                        img.save(os.path.join(
                            self.res_dir, subset_id, "images", img_path))
                
                except IOError:
                    os.remove(os.path.join(
                        self.res_dir, subset_id, "images", img_path))
                    file_not_found_cntr += 1

            if rot_cntr > 0:
                print(f"Number of images rotated: {rot_cntr}.\n")
            if file_not_found_cntr > 0:
                print(f"Deleted {file_not_found_cntr} images.\n")

    def install(self):
        subsets = ['train', 'valid']
        original = ['train2017', 'val2017']
        
        for origin, subset_id in zip(original, subsets):

            cntr = 0

            imgs_paths = os.listdir(os.path.join(
                self.res_dir, subset_id, "images"))

            for img_path in tqdm(
                iterable=imgs_paths,
                desc=f"Parsing {subset_id} subset for samples",
                total=len(imgs_paths),
                unit=" samples"
            ):
                img = Image.open(os.path.join(
                    self.res_dir, subset_id, "images", img_path))
                width, height = img.size

                try:
                    label = np.loadtxt(os.path.join(
                        self.res_dir,
                        "coco2017labels",
                        "coco", "labels", origin,
                        str(Path(img_path).resolve().stem) + '.txt')
                    )

                    if label.size != 0:
                        if label.ndim < 2:
                            label = label.reshape(1, -1)

                        classes = np.array(label[:, 0]).astype(int) + 1
                        x_center = np.array(label[:, 1] * width).reshape(1, -1)
                        y_center = np.array(
                            label[:, 2] * height).reshape(1, -1)
                        obj_width = np.array(
                            (label[:, 3] * width)).reshape(1, -1)
                        obj_height = np.array(
                            (label[:, 4] * height)).reshape(1, -1)

                        x1 = x_center - (obj_width / 2)
                        y1 = y_center - (obj_height / 2)
                        x2 = x_center + (obj_width / 2)
                        y2 = y_center + (obj_height / 2)

                        annotation_bbox = np.ceil(
                            np.vstack((x1, y1, x2, y2)).T)

                        annotation_sample = np.column_stack(
                            (classes.astype(str), annotation_bbox.astype(str)))

                        np.savetxt(
                            fname=os.path.join(
                                self.res_dir, subset_id, "labels", str(Path(img_path).resolve().stem) + '.txt'),
                            X=annotation_sample,
                            fmt='%s')

                except IOError:
                    cntr += 1
            
            if cntr > 0:
                print(f"Number of files not found: {cntr}.\n")
            else:
                print(f"All files were found.")

        shutil.rmtree(os.path.join(self.res_dir, "coco2017labels"))
        os.remove(os.path.join(self.res_dir, "coco2017labels.zip"))

    def driver(self):
        self.configure()
        self.download()
        self.extract()
        self.clean()
        self.install()
        self.preprocess()


url_train_imgs = "http://images.cocodataset.org/zips/train2017.zip"
url_valid_imgs = "http://images.cocodataset.org/zips/val2017.zip"
url_lbls = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"

res_dir = Path("../datasets/coco")

coco_installer = installCocoDataset(
    urls=[url_train_imgs, url_valid_imgs, url_lbls], res_dir=res_dir)
coco_installer.driver()


# TODO embed jit for faster transformations

import cv2
import torch
import torchvision
import numpy as np
import albumentations as A
import lib.transforms as T

from albumentations.pytorch import ToTensorV2


class DetectionPresetTrainTorchVision:
    def __init__(self, img_size: int = 640,  hflip_prob: float = 0.5):

        self.transform = T.Compose([
            T.Resize(img_size=img_size),
            T.RandomPhotometricDistort(),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])
    
    def __call__(self, img, target):
        return self.transform(img, target)
    

class DetectionPresetEvalTorchVision:
    def __init__(self, img_size: int = 640):

        self.transform = T.Compose([
            T.Resize(img_size=img_size),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

    def __call__(self, img, target):
        return self.transform(img, target)


class DetectionPresetTestTorchVision:
    def __init__(self, img_size: int = 640):

        self.transform = torchvision.transforms.Compose([
            T.ResizeImage(img_size=img_size),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

    def __call__(self, img):
        return self.transform(img)


class DetectionPresetImageOnlyTorchVision:
    def __init__(self, img_size: int = 640):
        self.transform = T.ResizeImage(img_size=img_size)

    def __call__(self, img):
        return self.transform(img)


class DetectionPresetTargetOnlyTorchVision:
    def __init__(self, img_size: int = 640):
        self.transform = T.ResizeTarget(img_size=img_size)

    def __call__(self, target, height: int, width: int):
        return self.transform(target=target, height=height, width=width)


class DetectionPresetTrainAlbumentations:
    """
    https://albumentations.ai/docs/examples/pytorch_classification/

    https://github.com/ultralytics/yolov5/blob/b8f979bafab6db020d86779b4b40619cd4d77d57/utils/augmentations.py#L25
    """

    def __init__(self, img_size: int = 640):

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, value=0, border_mode=cv2.BORDER_CONSTANT),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.01),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.01),
            A.RandomGamma(p=0.01),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.01),
            A.RandomBrightnessContrast(p=0.01),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, img, labels):
        res_augmented = self.transform(
            image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        img, labels = res_augmented['image'], np.array(
            [[c, *b] for c, b in zip(res_augmented['class_labels'], res_augmented['bboxes'])])
        return img, labels


class DetectionPresetEvalAlbumentations:
    def __init__(self, img_size: int = 640):
        
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, img, labels):
        res_augmented = self.transforms(
            image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        img, labels = res_augmented['image'], np.array(
            [[c, *b] for c, b in zip(res_augmented['class_labels'], res_augmented['bboxes'])])
        return img, labels


class DetectionPresetTestAlbumentations:
    def __init__(self, img_size: int = 640):

        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transforms(image=image)["image"]


class ScaleImage:
    def __init__(self, img_size: int = 640):
        
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT)
        ])
        
    def __call__(self, image):
        return self.transforms(image=image)["image"]


class ScaleTarget:
    def __init__(self, img_size: int = 640):

        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, labels, height: int, width: int):
        zero_img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        res_augmented = self.transforms(
            image=zero_img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        labels = np.array([[c, *b] for c, b in zip(res_augmented['class_labels'], res_augmented['bboxes'])])
        return labels

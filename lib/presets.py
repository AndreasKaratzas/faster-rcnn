
import cv2
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2


class DetectionPresetTrain:
    """
    
    https://albumentations.ai/docs/examples/pytorch_classification/

    https://github.com/ultralytics/yolov5/blob/b8f979bafab6db020d86779b4b40619cd4d77d57/utils/augmentations.py#L25
    """

    def __init__(self, img_size: int = 640):

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size),
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


class DetectionPresetEval:
    def __init__(self, img_size: int = 1280):
        
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, img, labels):
        res_augmented = self.transform(
            image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
        img, labels = res_augmented['image'], np.array(
            [[c, *b] for c, b in zip(res_augmented['class_labels'], res_augmented['bboxes'])])
        return img, labels


class DetectionPresetTest:
    def __init__(self, img_size: int = 1280):
        
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          value=0, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2()
        ])
        
    def __call__(self, image):
        return self.transforms(image=image)




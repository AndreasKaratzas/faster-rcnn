
import sys
import math
import time
import torch
import torchvision
from random import random

from coco.coco_utils import get_coco_api_from_dataset
from lib.utils import reduce_dict, warmup_lr_scheduler

from lib.visual import VisualTest
from coco.coco_eval import CocoEvaluator
from lib.metrics import MetricLogger, SmoothedValue


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def train(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    verbosity: int, 
    epoch: int, 
    log_filepath: str,
    confirm: bool = False,
    sample: float = 0.10,
    num_classes: int = 12
):
    model.train()
    metric_logger = MetricLogger(f_path=log_filepath, delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # test dataloader
    if confirm:
        if sample > 1.0 or sample < 0.0:
            raise ValueError(f"Option `sample` was not between [0, 1]. Input value was {sample}.")

        visualize = VisualTest(num_classes=num_classes)
        for images, targets in dataloader:
            if random() < sample:
                for image, target in zip(images, targets):
                    if target['boxes'].ndim < 1:
                        target['boxes'] = target['boxes'].unsqueeze(0)
                    
                    visualize.visualize(
                        img=image * 255, boxes=target['boxes'], labels=target['labels'])

    for images, targets in metric_logger.log_every(dataloader, verbosity, epoch + 1):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

@torch.no_grad()
def validate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    verbosity: int, 
    log_filepath: str,
    epoch: int
):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(f_path=log_filepath, delimiter="  ")

    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, log_filepath, epoch)
    
    for images, targets in metric_logger.log_every(dataloader, verbosity, -1):
        images = list(image.to(device) for image in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()       
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

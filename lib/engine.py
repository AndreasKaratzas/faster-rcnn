
import sys
import math
import time
import torch
import numpy as np
import torchvision
from apex import amp
from tqdm import tqdm
from random import random
from statistics import mean

from coco.coco_utils import get_coco_api_from_dataset
from lib.utils import reduce_dict, warmup_lr_scheduler, blockPrint, enablePrint

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
    tb_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    epoch: int, 
    log_filepath: str,
    writer: object,
    apex_activated: bool,
    sample: float = 0.01,
    num_classes: int = 12,
    no_visual: bool = False, 
    no_save: bool = True,
    res_dir: str = ''
):
    model.train()
    metric_logger = MetricLogger(f_path=log_filepath, delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # training dataloader
    if not no_visual and epoch == 0:
        if sample > 1.0 or sample < 0.0:
            raise ValueError(f"Option `sample` was not between [0, 1]. Input value was {sample}.")
        visualize = VisualTest(num_classes=num_classes, res_dir=res_dir)
        
        sample_num = 0
        grid = []

        for images, targets in tqdm(
            iterable=tb_dataloader, 
            total=len(tb_dataloader), 
            bar_format='{l_bar}{bar:35}{r_bar}{bar:-35b}',
            unit=' batches',
            desc='Configuring batch visualization in tensorboard'
        ):
            for image, target in zip(images, targets):

                if target['boxes'].ndim < 1:
                    target['boxes'] = target['boxes'].unsqueeze(0)
                
                sample_image = visualize.visualize(
                    img=image * 255, boxes=target['boxes'], labels=target['labels'], 
                    no_visual=True, no_save=no_save)

                grid.append(sample_image)

                if len(grid) == 4:
                    grid = torchvision.utils.make_grid(grid)
                    writer.add_image('batch_' + str(sample_num), grid, 0)
                    sample_num += 1
                    grid = []
    
    # Define loss accumulators for statistics
    loss_acc = []
    loss_classifier_acc = []
    loss_box_reg_acc = []
    loss_objectness_acc = []
    loss_rpn_box_reg_acc = []

    print(
        f"\n\n\t{'Epoch':10}{'gpu_mem':15}{'lr':10}{'loss':10}{'cls':10}{'box':10}{'obj':10}{'rpn':10}")
    with tqdm(total=len(dataloader), bar_format='{l_bar}{bar:35}{r_bar}{bar:-35b}') as pbar:
        for images, targets in metric_logger.log_every(dataloader, epoch + 1):
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

            blockPrint()
            optimizer.zero_grad()
            
            if apex_activated:
                with amp.scale_loss(losses, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()

            optimizer.step()
            enablePrint()
            
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            lr = optimizer.param_groups[0]["lr"]
            loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = metric_logger.get_metrics()
            
            loss_acc.append(loss)
            loss_classifier_acc.append(loss_classifier)
            loss_box_reg_acc.append(loss_box_reg)
            loss_objectness_acc.append(loss_objectness)
            loss_rpn_box_reg_acc.append(loss_rpn_box_reg)
 
            pbar.set_description(('%13s' + '%12s' + '%10.3g' + '%12.3g' + '%9.3g' + '%10.3g' * 3) % (
                f'{epoch + 1}/{epochs}', 
                f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G',
                lr, round(mean(loss_acc), 3), round(mean(loss_classifier_acc), 3), 
                round(mean(loss_box_reg_acc), 3), round(mean(loss_objectness_acc), 3), 
                round(mean(loss_rpn_box_reg_acc), 3)))
            
            pbar.update(1)
        
        pbar.close()

    return \
        metric_logger, lr, \
        mean(loss_acc), \
        mean(loss_classifier_acc), \
        mean(loss_box_reg_acc), \
        mean(loss_objectness_acc), \
        mean(loss_rpn_box_reg_acc)


@torch.no_grad()
def validate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    log_filepath: str,
    epoch: int
):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    blockPrint()
    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, log_filepath, epoch + 1)
    enablePrint()

    pbar = tqdm(dataloader, desc=f"{'               mAP@.5:.95':29}"
                                 f"{'mAP@.5':11}{'mAP@.75':11}"
                                 f"{'mAP@s':10}{'mAP@m':10}"
                                 f"{'mAP@l':9}{'Recall':6}",
                bar_format='{l_bar}{bar:35}{r_bar}{bar:-35b}')
    for images, targets in pbar:
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
        coco_evaluator.update(res)
    
    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()
    
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    print(('%25.3g' + '%10.3g' + '%12.3g' + '%9.3g' + '%10.3g' + '%10.3g' + '%10.3g') % (
        coco_evaluator.stats[0], coco_evaluator.stats[1], coco_evaluator.stats[2], coco_evaluator.stats[3],
        coco_evaluator.stats[4], coco_evaluator.stats[5], np.mean(coco_evaluator.stats[6:])))
    torch.set_num_threads(n_threads)

    val_metrics = {
        "mAP@.5:.95": coco_evaluator.stats[0],
        "mAP@.5": coco_evaluator.stats[1],
        "mAP@.75": coco_evaluator.stats[2],
        "mAP@s": coco_evaluator.stats[3],
        "mAP@m": coco_evaluator.stats[4],
        "mAP@l": coco_evaluator.stats[5],
        "Recall": np.mean(coco_evaluator.stats[6:])
    }

    return val_metrics

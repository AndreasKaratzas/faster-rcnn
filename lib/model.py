
import torch

from typing import Tuple, List

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone

from lib.utils import list2tup, flatten


def compile_backbone(
    backbone_name: str = 'resnet50',
    trainable_layers: int = 3
) -> torch.nn.Module:
    """Returns the backbone model."""
    if 'resnet' in backbone_name:
        backbone = resnet_fpn_backbone(backbone_name, trainable_layers=trainable_layers, pretrained=True)
    elif 'mobilenet' in backbone_name:
        backbone = mobilenet_backbone(backbone_name, trainable_layers=trainable_layers, pretrained=True)
    else:
        raise ValueError(
            f'Backbone model name option is invalid. Input value was {backbone_name}')
    return backbone


def compile_anchor(
    anchor_sizes: Tuple[Tuple[int]] = None, aspect_ratios: Tuple[Tuple[int]] = None
) -> AnchorGenerator:
    """Returns the anchor generator."""
    if anchor_sizes is None:
        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    else:
        aspect_ratios = ((tuple(flatten(aspect_ratios)), ) * len(anchor_sizes))
    
    return AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)


def compile_rpn(
    in_channels: int, num_anchors: int
) -> RPNHead:
    """Returns the RPN Head"""
    return RPNHead(in_channels=in_channels, num_anchors=num_anchors)


def compile_roi(
    featmap_names: List[str] = ["0"], output_size: int = 7, sampling_ratio: int = 2
) -> MultiScaleRoIAlign:
    """Returns the ROI Pooling"""
    return MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio)


def compile_predictor(
    in_channels: int, num_classes: int
) -> FastRCNNPredictor:
    """Returns the Predictor model"""
    return FastRCNNPredictor(
        in_channels=in_channels,
        num_classes=num_classes)


def compile_rcnn(
    backbone: torch.nn.Module,
    rpn_anchor_generator: AnchorGenerator,
    rpn_head: RPNHead,
    box_roi_pool: MultiScaleRoIAlign,
    box_predictor: FastRCNNPredictor,
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: List[float] = [0.485, 0.456, 0.406],
    image_std: List[float] = [0.229, 0.224, 0.225],
) -> FasterRCNN:
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""

    return FasterRCNN(
        backbone=backbone, 
        rpn_anchor_generator=rpn_anchor_generator, 
        rpn_head=rpn_head, 
        box_roi_pool=box_roi_pool, 
        box_predictor=box_predictor, 
        min_size=min_size, 
        max_size=max_size, 
        image_mean=image_mean, 
        image_std=image_std
    )


def configure_model(
    backbone_name: str,
    anchor_sizes: List[int],
    aspect_ratios: List[int],
    trainable_layers: int = 3,
    featmap_names: List[str] = ['0'],
    output_size: int = 7,
    sampling_ratio: int = 2,
    representation_size: int = 1024,
    num_classes: int = 2,
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: List[int] = [0.485, 0.456, 0.406],
    image_std: List[int] =[0.229, 0.224, 0.225]
) -> FasterRCNN:
    """Driver definition for Faster R-CNN model compilation"""
    backbone = compile_backbone(
        backbone_name=backbone_name, trainable_layers=trainable_layers)
    
    rpn_anchor_generator = compile_anchor(
        anchor_sizes=list2tup(anchor_sizes),
        aspect_ratios=list2tup(aspect_ratios)
    )

    rpn_head = compile_rpn(
        in_channels=backbone.out_channels,
        num_anchors=rpn_anchor_generator.num_anchors_per_location()[0]
    )

    box_roi_pool = compile_roi(
        featmap_names=featmap_names,
        output_size=output_size, 
        sampling_ratio=sampling_ratio
    )

    box_predictor = compile_predictor(
        in_channels=representation_size,
        num_classes=num_classes
    )

    model = compile_rcnn(
        backbone=backbone,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=box_roi_pool,
        box_predictor=box_predictor,
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std
    )

    return model

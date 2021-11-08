
import os
import json
import torch
import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from lib.visual import Visual
from lib.utils import get_transform
from lib.model import configure_model
from lib.dataloader import TestDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Faster R-CNN.')
    parser.add_argument('--model-checkpoint', type=str, help='Path to pretrained model.')
    parser.add_argument('--dataset', default='../data-faster', type=str, help='Path to dataset.')
    parser.add_argument('--batch-size', default=4, type=int, help='Batch size (default: 4).')
    parser.add_argument('--img-size', default=1280, type=int,
                        help='Minimum image size (default: 1280).')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='Number of data loading workers (default: 8).')
    parser.add_argument('--conf-threshold', default=0.50, type=float, help='Confidence threshold in prediction (default: 0.5).')
    parser.add_argument('--visualize', default=0.50, type=float,
                        help='Visualize a percentage of predictions (default: 0.5).')
    args = parser.parse_args()

    if not Path(args.model_checkpoint).is_file():
        raise ValueError(f"Path to pretrained model weights is invalid. Value parsed {args.model_checkpoint}.")
    if not Path(args.dataset).joinpath("test").is_dir():
        raise ValueError(
            f"Path to dataset is invalid. Value parsed {Path(args.dataset).joinpath('test')}.")

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilized:\t[{device}]\n")

    with open(Path(args.model_checkpoint).parent.absolute().parent.absolute().joinpath('CONFIG.json')) as f:
        data = json.load(f)

        args.backbone = data['model']['backbone']
        args.anchor_sizes = data['model']['anchors']
        args.aspect_ratios = data['model']['ratios']
        args.epochs = data['model']['epochs']
        
        args.num_classes = data['dataset']['classes']
        args.img_size = data['dataset']['img_size']
        
    # model init
    model = configure_model(
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes,
        aspect_ratios=args.aspect_ratios,
        num_classes=args.num_classes
    )

    # test dataset
    test_data = TestDataset(
        root_dir=os.path.join(args.dataset, "test"),
        transforms=get_transform(
            transform_class="test", 
            img_size=args.img_size)
    )

    # test dataloader
    dataloader_test = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    # load model to device
    model = model.to(device)

    # load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model checkpoint at {args.model_checkpoint} successfully")

    # test model
    visualize = Visual(model=model, root_dir=args.dataset,
                       device=device, conf_threshold=args.conf_threshold)
    visualize.test_model(dataloader=dataloader_test)

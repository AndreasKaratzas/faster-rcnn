
import os
import json
import math
import torch
import argparse
import datetime
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader

from lib.utils import collate_fn, get_transform
from lib.model import configure_model
from lib.engine import train, validate
from lib.plots import experiment_data_plots

from lib.elitism import EliteModel
from lib.dataloader import CustomDetectionDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Detection with Faster R-CNN.')
    parser.add_argument('--root-dir', default='./data', help='Root directory to output data.')
    parser.add_argument('--dataset', default='../data-faster', help='Path to dataset.')
    parser.add_argument('--img-size', default=640, type=int,  help='Minimum image size (default: 640).')
    parser.add_argument('--num-classes', default=12, type=int,  help='Number of classes in dataset including background.')
    parser.add_argument('--backbone', default='resnet50', help='Backbone CNN for Faster R-CNN (default: resnet50).')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size (default: 16).')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr).')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of total epochs to run (default: 100).')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N', help='Number of data loading workers (default: 8).')
    parser.add_argument('--lr', default=5e-3, type=float, help='Initial learning rate (default: 5e-3).')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum.')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4).')
    parser.add_argument('--lr-steps', default=[2000, 4000], nargs='+', type=int, help='Decrease lr every step-size epochs.')
    parser.add_argument('--lr-gamma', default=1e-2, type=float, help='Decrease lr by a factor of lr-gamma.')
    parser.add_argument('--verbosity', default=5, type=int, help='Terminal log frequency (default: 5).')
    parser.add_argument('--resume', type=str, default=None, help='Resume from given checkpoint. Expecting filepath to checkpoint.')
    parser.add_argument('--pretrained', default=True, help='Use pre-trained models (default: true).', action="store_true")
    parser.add_argument('--anchor-sizes', default=[4, 8, 16, 32, 128], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument('--aspect-ratios', default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nargs='+', type=int, help='Anchor ratios.')
    parser.add_argument('--start-epoch', default=0, type=int, help='Start epoch.')
    parser.add_argument('--conf-threshold', default=0.5, type=float, help='Prediction score threshold.')
    parser.add_argument('--trainable-layers', default=3,
                        type=int, help='Number of CNN backbone layers to train (min: 0, max: 5, default: 3).')
    args = parser.parse_args()

    # TODO describe directory formatting ['train', 'valid' and then 'images', 'labels']
    # Powershell:                     Dir | ren -NewName { $_.Name -replace ".png", ""}
    if not Path(args.root_dir).is_dir():
        raise ValueError(f"Root directory is invalid. Value parsed {args.root_dir}.")
    if not Path(args.dataset).is_dir():
        raise ValueError(f"Path to dataset is invalid. Value parsed {args.dataset}.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "images")):
        raise ValueError(f"Path to training image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "labels")):
        raise ValueError(f"Path to training label data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "valid", "images")):
        raise ValueError(f"Path to validation image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "valid", "labels")):
        raise ValueError(f"Path to validation label data does not exist.")

    if args.trainable_layers > 5 or args.trainable_layers < 0:
        raise ValueError(f"Number of CNN backbone trainable layers must be an integer defined between 0 and 5.")
    
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilized:\t[{device}]\n")

    # training dataset
    train_data = CustomDetectionDataset(
        root_dir=os.path.join(args.dataset, "train"),
        img_size=args.img_size,
        transforms=get_transform(
            transform_class="train",
            img_size=args.img_size)
    )

    # validation dataset
    val_data = CustomDetectionDataset(
        root_dir=os.path.join(args.dataset, "valid"),
        img_size=args.img_size,
        transforms=get_transform(
            transform_class="valid",
            img_size=args.img_size
        )
    )

    print(
        f'Training Faster R-CNN with model backbone {args.backbone} with {args.trainable_layers} trainable layers, anchor'
        f' sizes {args.anchor_sizes} and aspect ratios {args.aspect_ratios} for {args.epochs} epochs.')
    print(f'Length of train data {len(train_data)}')
    print(f'Length of validation data {len(val_data)}')

    # training dataloader
    dataloader_train = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # validation dataloader
    dataloader_valid = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # custom model init
    model = configure_model(
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes,
        aspect_ratios=args.aspect_ratios,
        num_classes=args.num_classes,
        trainable_layers=args.trainable_layers
    )
 
    optimizer = optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad], 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    # load model to device
    model = model.to(device)

    # setup output paths
    datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_save_dir = Path(args.root_dir) / datetime_tag / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir = Path(args.root_dir) / datetime_tag / "log"
    log_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir = Path(args.root_dir) / datetime_tag / "plots"
    plots_save_dir.mkdir(parents=True, exist_ok=True)
    config_save_dir = Path(args.root_dir) / datetime_tag / "CONFIG.json"

    with open(config_save_dir, "w") as f:
        data = {}

        data['model'] = {
            'backbone': args.backbone,
            'anchors': args.anchor_sizes,
            'ratios': args.aspect_ratios,
            'epochs': args.epochs,
            'checkpoint': args.resume,
            'start': args.start_epoch,
            'trainable': args.trainable_layers
        }

        data['dataset'] = {
            'classes': args.num_classes,
            'img_size': args.img_size,
            'directory': args.dataset
        }

        json.dump(data, f)

    log_save_dir_train = log_save_dir / "training"
    log_save_dir_validation = log_save_dir / "validation"

    # initialize lr scheduler
    if args.lr_scheduler.lower() == 'multisteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler.lower() == 'cosineannealinglr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler.lower()))
    
    # load checkpoint
    if args.resume:
        if not Path(args.resume).is_file():
            raise ValueError(f"Checkpoint filepath was not found. Tried to access {args.resume}.")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"Training model from checkpoint {args.resume}. Starting from epoch {args.start_epoch}.")
    
    # prepare training logger
    with open(log_save_dir_train, "w") as f:
        f.write(
            f"{'Epoch':>8}{'TimeDelta':>10}{'LearningRate':>15}{'Loss':>15}"
            f"{'LossClassifier':>15}{'LossBoxReg':>15}{'LossObjectness':>15}"
            f"{'LossRpnBoxReg':>15}{'Cuda':>10}\n"
        )
    
    # prepare validation logger
    with open(log_save_dir_validation, "w") as f:
        f.write(
            f"{'Epoch':>8}{'Title':>20}{'IoU':>15}{'Area':>8}"
            f"{'MaxDets':>8}{'Value':>8}\n"
        )
    
    # reconfigure verbosity
    if args.verbosity is not None:
        args.verbosity = int(math.ceil(
            (len(train_data) / args.batch_size) / args.verbosity))
    else:
        args.verbosity = int(math.ceil(
            (len(train_data) / args.batch_size) / 5))
    
    # configure verbosity while testing model 
    test_verbosity = int(math.ceil(
        (len(val_data) * args.verbosity * args.batch_size) / len(train_data)))

    # initialize best model criterion
    elite_model_criterion = EliteModel(data_dir=log_save_dir)

    # start fitting the model
    for epoch in range(args.start_epoch, args.epochs):
        
        train_logger = train(model=model, optimizer=optimizer, dataloader=dataloader_train, device=device,
                             verbosity=args.verbosity, epoch=epoch, log_filepath=log_save_dir_train, num_classes=args.num_classes)
        train_logger.export_data()
        validate(model=model, dataloader=dataloader_valid, device=device,
                 verbosity=test_verbosity, log_filepath=log_save_dir_validation, epoch=epoch)

        lr_scheduler.step()
        elite_model_criterion.calculate_metrics(epoch=epoch)
        
        if elite_model_criterion.evaluate_model():
            # save best model to disk
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(model_save_dir, 'best.pt'))
        
        # save last model to disk
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, os.path.join(model_save_dir, 'last.pt'))

    experiment_data_plots(root_dir=log_save_dir, out_dir=plots_save_dir)

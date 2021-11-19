
import argparse
import datetime
import json
import os
import warnings
import platform
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.autoanchor import autoanchors
from lib.cacher import CustomCachedDetectionDataset
from lib.elitism import EliteModel
from lib.engine import train, validate
from lib.model import configure_model
from lib.plots import experiment_data_plots
from lib.utils import collate_fn, get_transform

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Detection with Faster R-CNN.')
    parser.add_argument('--root-dir', default='./data',
                        help='Root directory to output data.')
    parser.add_argument('--dataset', default='../data-faster',
                        help='Path to dataset.')
    parser.add_argument('--project', default='AITrain', type=str,
                        help='Name of project.')
    parser.add_argument('--img-size', default=640, type=int,
                        help='Image size (default: 640).')
    parser.add_argument('--num-classes', default=12, type=int,
                        help='Number of classes in dataset including background.')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Backbone CNN for Faster R-CNN (default: resnet50).')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size (default: 16).')
    parser.add_argument('--lr-scheduler', default="multisteplr", type=str,
                        help='the lr scheduler (default: multisteplr).')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='Number of total epochs to run (default: 100).')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8).')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Initial learning rate (default: 1e-3).')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='Momentum.')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4).')
    parser.add_argument('--lr-steps', default=[16000, 22000], nargs='+',
                        type=int, help='Decrease lr every step-size epochs.')
    parser.add_argument('--lr-gamma', default=1e-2, type=float,
                        help='Decrease lr by a factor of lr-gamma.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint. Expecting filepath to checkpoint.')
    parser.add_argument('--cache', default=True, action='store_true', type=bool,
                        help='Cache the images found in the dataset (default: True).')
    parser.add_argument('--pretrained', default=True, type=bool,
                        help='Use pre-trained models (default: true).', action="store_true")
    parser.add_argument(
        '--anchor-sizes', default=[4, 8, 16, 32, 128], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument('--aspect-ratios', default=[
                        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nargs='+', type=int, help='Anchor ratios.')
    parser.add_argument('--no-autoanchor', default=False, action='store_true',
                        help='Disable anchor recommendation software.')
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='Start epoch.')
    parser.add_argument('--conf-threshold', default=0.5,
                        type=float, help='Prediction score threshold.')
    parser.add_argument('--trainable-layers', default=3,
                        type=int, help='Number of CNN backbone layers to train (min: 0, max: 5, default: 3).')
    parser.add_argument('--no-visual', default=True, action='store_true', type=bool,
                        help='Disable visualization software in test mode.')
    parser.add_argument('--no-save', default=False, action='store_true', type=bool,
                        help='Disable results export software.')
    parser.add_argument('--no-threading-linux', default=True, action='store_true', type=bool,
                        help='Disable multithreading library in Linux due to possible race conditions.')
    args = parser.parse_args()

    # TODO describe directory formatting ['train', 'valid' and then 'images', 'labels']
    if not Path(args.root_dir).is_dir():
        raise ValueError(
            f"Root directory is invalid. Value parsed {args.root_dir}.")
    if not Path(args.dataset).is_dir():
        raise ValueError(
            f"Path to dataset is invalid. Value parsed {args.dataset}.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "images")):
        raise ValueError(f"Path to training image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "train", "labels")):
        raise ValueError(f"Path to training label data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "valid", "images")):
        raise ValueError(f"Path to validation image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset, "valid", "labels")):
        raise ValueError(f"Path to validation label data does not exist.")

    if args.trainable_layers > 5 or args.trainable_layers < 0:
        raise ValueError(
            f"Number of CNN backbone trainable layers must be an integer defined between 0 and 5.")
    
    # check platform and reconfigure number of workers
    if platform.system() == "Linux" and args.no_threading_linux:
        # RuntimeError: received 0 items of ancdata
        args.num_workers = 1
        print(f"WARNING:"
            f"\n\tOS family is Linux."
            f"\n\tLibrary `multithreading` in Python might not function well."
            f"\n\tSetting number of workers equal to {args.num_workers}.\n")

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilized:\t[{device}]\n")

    # training dataset
    # train_data = CustomDetectionDataset(
    #     root_dir=os.path.join(args.dataset, "train"),
    #     transforms=get_transform(
    #         transform_class="train",
    #         img_size=args.img_size)
    # )

    # training dataset
    train_data = CustomCachedDetectionDataset(
        root_dir=os.path.join(args.dataset, "train"),
        num_threads=args.num_workers,
        batch_size=args.batch_size,
        img_size=args.img_size,
        cache_images_flag=args.cache,
        transforms=get_transform(
            transform_class="train",
            img_size=args.img_size)
    )

    # validation dataset
    val_data = CustomCachedDetectionDataset(
        root_dir=os.path.join(args.dataset, "valid"),
        num_threads=args.num_workers,
        batch_size=args.batch_size,
        img_size=args.img_size,
        cache_images_flag=args.cache,
        transforms=get_transform(
            transform_class="valid",
            img_size=args.img_size
        )
    )

    # training dataloader
    # TODO https://github.com/pytorch/vision/blob/aedd39792d07af58e55ec028ed344d63debbd281/references/detection/train.py#L170
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

    # autoanchor software
    if not args.no_autoanchor:
        args.anchor_sizes, args.aspect_ratios = autoanchors(
            dataloader=dataloader_valid)
        print(
            f"\nSetting the following hyperparameters to the recommended values:\n\t"
            f"Anchor sizes:\t{args.anchor_sizes}\n\tAspect ratios:\t{args.aspect_ratios}\n\n"
            f"\tTo disable the automated anchor software, pass the `--no-autoanchor` option.\n")

    print(
        f'Training Faster R-CNN for {args.epochs} epoch(s) with model backbone {args.backbone} with:\n'
        f'\t{args.trainable_layers}\t\t\t\ttrainable layer(s)\n'
        f'\t{args.anchor_sizes}\tanchor sizes and\n'
        f'\t{args.aspect_ratios}\taspect ratios\n\nDataset stats:\n'
        f'\tLength of train data:\t\t{len(train_data):5d}\n'
        f'\tLength of validation data:\t{len(val_data):5d}\n\n\n')

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
    gt_save_dir = Path(args.root_dir) / datetime_tag / "ground_truth"
    gt_save_dir.mkdir(parents=True, exist_ok=True)
    config_save_dir = Path(args.root_dir) / datetime_tag / "CONFIG.json"

    with open(config_save_dir, "w") as f:
        data = {}

        data['model'] = {
            'backbone': args.backbone,
            'anchors': ['{:.2f}'.format(x) for x in args.anchor_sizes],
            'ratios': ['{:.2f}'.format(x) for x in args.aspect_ratios],
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

    log_save_dir_train = log_save_dir / "training.txt"
    log_save_dir_validation = log_save_dir / "validation.txt"

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
            raise ValueError(
                f"Checkpoint filepath was not found. Tried to access {args.resume}.")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(
            f"Training model from checkpoint {args.resume}. Starting from epoch {args.start_epoch}.")

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

    # initialize best model criterion
    elite_model_criterion = EliteModel(data_dir=log_save_dir)

    # start fitting the model
    for epoch in range(args.start_epoch, args.epochs):

        train_logger = train(model=model, optimizer=optimizer, dataloader=dataloader_train, device=device, epochs=args.epochs,
                             epoch=epoch, log_filepath=log_save_dir_train, num_classes=args.num_classes,
                             no_visual=args.no_visual, no_save=args.no_save, res_dir=gt_save_dir)
        train_logger.export_data()
        validate(model=model, dataloader=dataloader_valid, device=device,
                 log_filepath=log_save_dir_validation, epoch=epoch)

        lr_scheduler.step()
        elite_model_criterion.calculate_metrics(epoch=epoch + 1)

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

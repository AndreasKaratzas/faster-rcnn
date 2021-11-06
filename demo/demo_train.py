
import os
import sys
import math
import torch
import argparse
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(1, '../')

from lib.elitism import EliteModel
from lib.dataloader import PennFudanDataset

from lib.utils import collate_fn
from lib.model import configure_model
from lib.engine import train, validate
from lib.plots import experiment_data_plots
from lib.transformation import get_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch DEMO training script with Faster R-CNN.')
    parser.add_argument('--root-dir', default='../data',
                        help='Root directory to output data.')
    parser.add_argument('--dataset', default='../data/PennFudanPed/Train',
                        help='Path to dataset.')
    parser.add_argument('--img-size', default=800,
                        type=int,  help='Image size.')
    parser.add_argument('--num-classes', default=2, type=int,
                        help='Number of classes in dataset.')
    parser.add_argument('--backbone', default='resnet50',
                        help='Backbone CNN for Faster R-CNN.')
    parser.add_argument('--batch-size', default=2,
                        type=int, help='Batch size.')
    parser.add_argument('--lr-scheduler', default="multisteplr",
                        help='the lr scheduler (default: multisteplr).')
    parser.add_argument('--epochs', default=20, type=int,
                        metavar='N', help='Number of total epochs to run.')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4).')
    parser.add_argument('--lr', default=4e-3, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='Momentum.')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4).')
    parser.add_argument(
        '--lr-steps', default=[16, 22], nargs='+', type=int, help='Decrease lr every step-size epochs.')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='Decrease lr by a factor of lr-gamma.')
    parser.add_argument('--verbosity', type=int, default=5,
                        help='Terminal log frequency.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint. Expecting filepath to checkpoint.')
    parser.add_argument('--data-augmentation', default="hflip",
                        help='Data augmentation policy (default: hflip).')
    parser.add_argument('--pretrained', default=True,
                        help='Use pre-trained models.', action="store_true")
    parser.add_argument(
        '--anchor-sizes', default=[32, 64, 128, 256, 512], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument(
        '--aspect-ratios', default=[0.5, 1.0, 2.0], nargs='+', type=int, help='Anchor ratios.')
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='Start epoch.')
    parser.add_argument('--val-samples', default=20, metavar='N',
                        type=int, help='Number of samples for validation purposes.')
    args = parser.parse_args()

    if not Path(args.root_dir).is_dir():
        raise ValueError(
            f"Root directory is invalid. Value parsed {args.root_dir}.")
    if not Path(args.dataset).is_dir():
        raise ValueError(
            f"Path to dataset is invalid. Value parsed {args.dataset}.")

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader training
    train_data = PennFudanDataset(
        args.dataset, get_transform(True, args.data_augmentation))

    # dataloader validation
    val_data = PennFudanDataset(
        args.dataset, get_transform(False, args.data_augmentation))

    # split the dataset in training and validation set
    indices = torch.randperm(len(train_data)).tolist()

    # define training and validation data
    train_data = Subset(train_data, indices[:-args.val_samples])
    val_data = Subset(val_data, indices[-args.val_samples:])

    print(
        f'Training Faster R-CNN with model backbone {args.backbone} and anchor'
        f' sizes {args.anchor_sizes} for {args.epochs} epochs.')
    
    print(f'Length of train data {len(train_data)}')
    print(f'Length of validation data {len(val_data)}')

    # define training and validation data loaders
    dataloader_train = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn)

    dataloader_valid = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn)

    # custom model init
    model = configure_model(
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes,
        aspect_ratios=args.aspect_ratios,
        min_size=args.img_size
    )

    # define optimizer
    optimizer = optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # load model to device
    model = model.to(device)

    # reconfigure verbosity
    if args.verbosity is not None:
        args.verbosity = int(math.ceil(
            (len(train_data) / args.batch_size) / args.verbosity))
    else:
        args.verbosity = int(math.ceil(
            (len(train_data) / args.batch_size) / 5))

    # setup output paths
    model_save_dir = Path(args.root_dir) / 'demo' / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir = Path(args.root_dir) / 'demo' / "log"
    log_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir = Path(args.root_dir) / 'demo' / "plots"
    plots_save_dir.mkdir(parents=True, exist_ok=True)
    config_save_dir = Path(args.root_dir) / 'demo' / "CONFIG"


    with open(config_save_dir, "w") as f:
        f.write(
            f'Training Faster R-CNN with model backbone {args.backbone} and anchor'
            f' sizes {args.anchor_sizes} for {args.epochs} epochs.\n'
        )
        f.write(
            f"Training model from checkpoint {args.resume}. Starting from epoch {args.start_epoch}.\n"
        )
        f.write(
            f"Dataset dir {args.dataset}.\n"
        )

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

    # configure verbosity while testing model
    test_verbosity = int(math.ceil(
        (len(val_data) * args.verbosity * args.batch_size) / len(train_data)))

    # initialize best model criterion
    elite_model_criterion = EliteModel(data_dir=log_save_dir)

    # start fitting the model
    for epoch in range(args.start_epoch, args.epochs):

        train_logger = train(model=model, optimizer=optimizer, dataloader=dataloader_train, device=device,
                             verbosity=args.verbosity, epoch=epoch, log_filepath=log_save_dir_train)
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

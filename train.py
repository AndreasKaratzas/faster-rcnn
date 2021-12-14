
import argparse
import datetime
import json
import copy
import os
import warnings
import platform
from pathlib import Path
import torch

APEX_NOT_INSTALLED = False
try:
    from apex import amp
except ImportError:
    APEX_NOT_INSTALLED = True

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity

from torch.utils.tensorboard import SummaryWriter
from lib.autoanchor import autoanchors
from lib.cacher import CustomCachedDetectionDataset
from lib.elitism import EliteModel
from lib.nvidia import cuda_check
from lib.engine import train, validate
from lib.model import configure_model
from lib.plots import experiment_data_plots
from lib.utils import collate_fn, get_transform, weight_histograms, TraceWrapper

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
    parser.add_argument('--batch-size', default=4, type=int,
                        help='Batch size (default: 16).')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='Number of total epochs to run (default: 100).')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8).')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='Initial learning rate (default: 5e-3).')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='Momentum.')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4).')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint. Expecting filepath to checkpoint.')
    parser.add_argument('--cache', default=False, action='store_true',
                        help='Cache the images found in the dataset (default: True).')
    parser.add_argument('--profiling', default=False, action='store_true',
                        help='Profile the training loop (default: False).')
    parser.add_argument('--prof-settings', default=[
                        5, 5, 10, 3], nargs='+', type=int,
                        help=f'Profiling settings. The order is:\n'
                             f'1. wait (default: 5)\n'
                             f'2. warmup (default: 5)\n'
                             f'3. active (default: 10)\n'
                             f'4. repeat (default: 3)')
    parser.add_argument('--pretrained', default=True,
                        help='Use pre-trained models (default: true).', action="store_true")
    parser.add_argument(
        '--anchor-sizes', default=[4, 8, 16, 32, 128], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument('--aspect-ratios', default=[
                        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nargs='+', type=int, 
                        help='Anchor ratios.')
    parser.add_argument('--no-autoanchor', default=False, action='store_true',
                        help='Disable anchor recommendation software.')
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='Start epoch.')
    parser.add_argument('--conf-threshold', default=0.5,
                        type=float, help='Prediction score threshold (default: 0.5).')
    parser.add_argument('--trainable-layers', default=3,
                        type=int, help=f'Number of CNN backbone layers to train '
                                       f'(min: 0, max: 5, default: 3).')
    parser.add_argument('--no-mixed-precision', default=False, action='store_true',
                        help='Disable mixed precision for model training (default: False).')
    parser.add_argument('--no-visual', default=False, action='store_true',
                        help='Disable visualization of randomly selected samples in tensorboard (default: False).')
    parser.add_argument('--no-model-graph', default=True, action='store_true',
                        help='Disable visualization of model as a graph in tensorboard (default: True).')
    parser.add_argument('--no-save', default=True, action='store_true',
                        help='Disable results export software (default: True).')
    parser.add_argument('--no-threading-linux', default=True, action='store_true',
                        help='Disable multithreading library in Linux due to possible race conditions (default: True).')
    parser.add_argument('--no-onnx', default=False, action='store_true',
                        help='Disable model export in ONNX format (default: False).')
    parser.add_argument('--generate-script-module', default=True, action='store_true',
                        help='Use `torch.jit.trace` to generate a `torch.jit.ScriptModule` via tracing (default: True).')
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

    if device == torch.device('cuda'):
        cuda_arch = cuda_check()

        if not 'Turing' in cuda_arch and not 'Volta' in cuda_arch:

            if APEX_NOT_INSTALLED:
                print(f"Found NVIDIA GPU of {cuda_arch} Architecture.")
                print(f"Disabling apex software for mixed-precision model training.")
                args.no_mixed_precision = True

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
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    dataloader_tb = None
    if not args.no_visual:
        tb_idx = torch.randperm(len(train_data))[:int(len(train_data) * 1e-1)]
        tb_sampler = torch.utils.data.SubsetRandomSampler(tb_idx)
        dataloader_tb = DataLoader(
            dataset=train_data, 
            batch_size=args.batch_size * 2, 
            shuffle=False, 
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            sampler=tb_sampler
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
        f'\tLength of validation data:\t{len(val_data):5d}\n\n')

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
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # load checkpoint
    if args.resume:
        if not Path(args.resume).is_file():
            raise ValueError(
                f"Checkpoint filepath was not found. Tried to access {args.resume}.")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        if type(checkpoint['amp']) is not None:
            amp.load_state_dict(checkpoint['amp'])
        
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
    
    # initialize tensorboard instance
    writer = SummaryWriter(log_save_dir, comment=args.project)

    if not args.no_model_graph:
        tb_model = TraceWrapper(model)
        tb_model.eval()
        writer.add_graph(
            model=tb_model,
            input_to_model=torch.rand([3, args.img_size, args.img_size]).unsqueeze(0), 
            verbose=False, 
            use_strict_trace=True
        )

    # load model to device
    model = model.to(device)

    if not args.no_mixed_precision:
        # Wrap the model
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # profile the model
    prof = None
    if args.profiling:
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.prof_settings[0],
                warmup=args.prof_settings[1],
                active=args.prof_settings[2],
                repeat=args.prof_settings[3]),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                log_save_dir),
            use_cuda=True if device == torch.device("cuda") else False,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            record_shapes=True,
            with_stack=True)

        prof.start()

    # start fitting the model
    for epoch in range(args.start_epoch, args.epochs):

        if (epoch >= (
            args.prof_settings[0] + 
            args.prof_settings[1] + 
            args.prof_settings[2]) * 
            args.prof_settings[3]
        ) and args.profiling:
            prof.stop()

        train_logger, lr, loss_acc, loss_classifier_acc, loss_box_reg_acc, loss_objectness_acc, loss_rpn_box_reg_acc = train(
            model=model, optimizer=optimizer, dataloader=dataloader_train, device=device, epochs=args.epochs,
            epoch=epoch, log_filepath=log_save_dir_train, writer=writer, num_classes=args.num_classes, apex_activated=not APEX_NOT_INSTALLED,
            no_visual=args.no_visual, no_save=args.no_save, res_dir=gt_save_dir, tb_dataloader=dataloader_tb)
        train_logger.export_data()

        writer.add_scalar('lr/train', lr, epoch)
        writer.add_scalar('loss/train', loss_acc, epoch)
        writer.add_scalar('loss_classifier/train', loss_classifier_acc, epoch)
        writer.add_scalar('loss_box_reg/train', loss_box_reg_acc, epoch)
        writer.add_scalar('loss_objectness/train', loss_objectness_acc, epoch)
        writer.add_scalar('loss_rpn_box_reg/train',
                          loss_rpn_box_reg_acc, epoch)

        val_metrics = validate(model=model, dataloader=dataloader_valid, device=device,
                               log_filepath=log_save_dir_validation, epoch=epoch)

        writer.add_scalar('mAP@.5:.95/validation',
                          val_metrics.get("mAP@.5:.95"), epoch)
        writer.add_scalar('mAP@.5/validation',
                          val_metrics.get("mAP@.5"), epoch)
        writer.add_scalar('mAP@.75/validation',
                          val_metrics.get("mAP@.75"), epoch)
        writer.add_scalar('mAP@s/validation',
                          val_metrics.get("mAP@s"), epoch)
        writer.add_scalar('mAP@m/validation',
                          val_metrics.get("mAP@m"), epoch)
        writer.add_scalar('mAP@l/validation',
                          val_metrics.get("mAP@l"), epoch)
        writer.add_scalar('Recall/validation',
                          val_metrics.get("Recall"), epoch)

        # Visualize weight histograms
        weight_histograms(writer, epoch, model)

        lr_scheduler.step(val_metrics.get("mAP@.5"))
        elite_model_criterion.calculate_metrics(epoch=epoch + 1)

        if elite_model_criterion.evaluate_model():
            # save best model to disk
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'amp': amp.state_dict() if not args.no_mixed_precision else None
            }, os.path.join(model_save_dir, 'best.pt'))

            if not args.no_onnx:
                onnx_model = copy.deepcopy(model)
                onnx_model.to('cpu')

                # Export the model
                torch.onnx.export(
                    # model being run
                    onnx_model,
                    # model input (or a tuple for multiple inputs)
                    torch.rand([3, args.img_size, args.img_size]).unsqueeze(0),
                    # where to save the model (can be a file or file-like object)
                    os.path.join(model_save_dir, 'best.onnx'),
                    # store the trained parameter weights inside the model file
                    export_params=True,
                    # the ONNX version to export the model to
                    opset_version=10,
                    # whether to execute constant folding for optimization
                    do_constant_folding=True,
                    # the model's input names
                    input_names=['input'],
                    # the model's output names
                    output_names=['output'],
                    # variable length axes
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    },
                    verbose=True
                )
            if args.generate_script_module:
                model_copy = copy.deepcopy(model)
                model_copy.to('cpu')

                # An example input provided to the defined model
                example = torch.rand(
                    [3, args.img_size, args.img_size]).unsqueeze(0)

                # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
                traced_script_module = torch.jit.trace(model_copy, example)

                # Serializing script module to a file
                traced_script_module.save(os.path.join(model_save_dir, 'traced_best.pt'))

        # save last model to disk
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'amp': amp.state_dict() if not args.no_mixed_precision else None
        }, os.path.join(model_save_dir, 'last.pt'))

        if not args.no_onnx:
            onnx_model = copy.deepcopy(model)
            onnx_model.to('cpu')

            # Export the model
            torch.onnx.export(
                # model being run
                onnx_model,
                # model input (or a tuple for multiple inputs)
                torch.rand([3, args.img_size, args.img_size]).unsqueeze(0),
                # where to save the model (can be a file or file-like object)
                os.path.join(model_save_dir, 'last.onnx'),
                # store the trained parameter weights inside the model file
                export_params=True,
                # the ONNX version to export the model to
                opset_version=10,
                # whether to execute constant folding for optimization
                do_constant_folding=True,
                # the model's input names
                input_names=['input'],
                # the model's output names
                output_names=['output'],
                # variable length axes
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=True
            )
        
        if args.generate_script_module:
            model_copy = copy.deepcopy(model)
            model_copy.to('cpu')

            # An example input provided to the defined model
            example = torch.rand(
                [3, args.img_size, args.img_size]).unsqueeze(0)

            # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
            traced_script_module = torch.jit.trace(model_copy, example)

            # Serializing script module to a file
            traced_script_module.save(os.path.join(
                model_save_dir, 'traced_last.pt'))

        if (epoch < (
            args.prof_settings[0] + 
            args.prof_settings[1] + 
            args.prof_settings[2]) * 
            args.prof_settings[3]
        ) and args.profiling:
            prof.step()

    experiment_data_plots(root_dir=log_save_dir, out_dir=plots_save_dir)
    writer.close()

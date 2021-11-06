
import torch
import argparse

from pathlib import Path
from lib.visual import Visual
from lib.model import configure_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Faster R-CNN.')
    parser.add_argument('--model-checkpoint', type=str, help='Path to pretrained model.')
    parser.add_argument('--dataset', type=str, help='Path to dataset.')
    parser.add_argument('--img-size', default=800, type=int, help='Image size.')
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size.')
    parser.add_argument('--num-classes', default=2, type=int, help='Number of classes in dataset.')
    parser.add_argument('--backbone', default='resnet101', help='Backbone CNN for Faster R-CNN.')
    parser.add_argument('--num-workers', default=1, type=int, metavar='N', help='Number of data loading workers (default: 1).')
    parser.add_argument('--conf-threshold', default=0.50, type=float, help='Confidence threshold in prediction.')
    parser.add_argument('--visualize', default=0.40, type=float, help='Visualize a percentage of predictions.')
    parser.add_argument('--anchor-sizes', default=[32, 64, 128, 256, 512], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument('--aspect-ratios', default=[0.5, 1.0, 2.0], nargs='+', type=int, help='Anchor ratios.')
    parser.add_argument('--raw-root-dir', action='store_true', help='Pass this option if there is no `images` directory in the `dataset` filepath.')
    args = parser.parse_args()

    if not Path(args.model_checkpoint).is_file():
        raise ValueError(f"Path to pretrained model weights is invalid. Value parsed {args.model_checkpoint}.")
    if not Path(args.dataset).is_dir():
        raise ValueError(f"Path to dataset is invalid. Value parsed {args.dataset}.")

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model init
    model = configure_model(
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes,
        aspect_ratios=args.aspect_ratios,
        min_size=args.img_size
    )

    # load model to device
    model = model.to(device)

    # load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model checkpoint at {args.model_checkpoint} successfully")

    # test model
    visualize = Visual(model=model, root_dir=args.dataset,
                       device=device, conf_threshold=args.conf_threshold, 
                       batch_size=args.batch_size, raw_root_dir=args.raw_root_dir)
    visualize.build_batch()
    visualize.test_model()

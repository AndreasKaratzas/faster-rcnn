
import os
import re
import argparse
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def export_training_plots(root_dir: str, out_dir: str, dpi: int = 300):
    training_res_data = pd.read_csv(os.path.join(
        root_dir, 'training'), delim_whitespace=True)

    train_res_preprocessed = training_res_data.groupby(
        by="Epoch", dropna=False).mean()

    for column in train_res_preprocessed.columns:
        plt.figure()
        plt.tight_layout()
        plt.title(column)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(train_res_preprocessed[column])
        plt.savefig(os.path.join(out_dir, slugify(
            column.strip()) + '.jpg'), dpi=dpi)


def export_validation_plots(root_dir: str, out_dir: str, dpi: int = 300):
    validation_res_data = pd.read_csv(os.path.join(
        root_dir, 'validation'), delim_whitespace=True)

    validation_res_data = validation_res_data.astype({'MaxDets': 'str'})
    
    validation_res_data["Key"] = validation_res_data["Title"].map(lambda x: x.lstrip(
        'Average')) + ' ' + validation_res_data["IoU"] + ' ' + validation_res_data["Area"] + ' ' + validation_res_data["MaxDets"]
    
    column_values = validation_res_data[["Key"]].values.ravel()
    unique_values = pd.unique(column_values)

    val_res_preprocessed = []
    for value in unique_values:
        val_res_preprocessed.append(
            validation_res_data[validation_res_data["Key"] == value])
    
    for validation_metric in val_res_preprocessed:
        plt_title = validation_metric[["Key"]].values.ravel()
        plt_title = pd.unique(plt_title)
        
        plt.figure()
        plt.tight_layout()
        plt.title(plt_title.item())
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(validation_metric.Epoch, validation_metric.Value)
        plt.savefig(os.path.join(
            out_dir, slugify(plt_title.item().strip()) + '.jpg'), dpi=dpi)


def experiment_data_plots(root_dir: str, out_dir: str):
    export_training_plots(root_dir, out_dir)
    export_validation_plots(root_dir, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creating plots after training a Faster R-CNN model.')
    parser.add_argument('--root-dir', default='../data/demo/log', type=str,
                        help='Root directory of input data.')
    parser.add_argument('--out-dir', default='../data/demo/plots', type=str,
                        help='Directory to output plots.')
    args = parser.parse_args()

    if not Path(args.root_dir).is_dir():
        raise ValueError(
            f"Root directory is invalid. Value parsed {args.root_dir}.")
    if not Path(args.out_dir).is_dir():
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.isfile(os.path.join(args.root_dir, 'training')):
        raise ValueError(
            f"Training data file not found. Tried to access {os.path.join(args.root_dir, 'training')}.")
    if not os.path.isfile(os.path.join(args.root_dir, 'validation')):
        raise ValueError(
            f"Validation data file not found. Tried to access {os.path.join(args.root_dir, 'validation')}.")

    experiment_data_plots(root_dir=args.root_dir, out_dir=args.out_dir)

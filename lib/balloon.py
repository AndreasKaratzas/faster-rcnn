
import os
import json
import errno
import itertools
import numpy as np
import pandas as pd


def export_labels(root_dir: str):
    """Transforms `Balloon` dataset into Faster R-CNN standard format"""

    labels_dir = os.path.join(root_dir, "labels")
    
    if not os.path.exists(labels_dir):
        try:
            os.makedirs(labels_dir, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError(e)

    json_file = os.path.join(root_dir, "via_region_data.json")
    with open(json_file) as f:
        images_annotations = json.load(f)

    for _, v in images_annotations.items():

        filename = os.path.splitext(v["filename"])[0]

        annotations = v["regions"]

        faster_rcnn_df = pd.DataFrame(columns=['X1', 'Y1', 'X2', 'Y2'])
        for _, anno in annotations.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            faster_rcnn_df = faster_rcnn_df.append(
                {
                    'X1': np.min(px), 
                    'Y1': np.min(py), 
                    'X2': np.max(px), 
                    'Y2': np.max(py)
                }, ignore_index=True)

        faster_rcnn_df = faster_rcnn_df.astype(
            {
                'X1': 'float64',
                'Y2': 'float64',
                'X2': 'float64',
                'Y2': 'float64'
            }
        )

        faster_rcnn_df.to_csv(os.path.join(
            labels_dir, filename + '.txt'), header=None, index=None, sep=' ', mode='w+')


def move_images(src_dir: str, dest_dir: str):

    if not os.path.exists(src_dir):
        raise ValueError(f"Directory {src_dir} does not exist")
    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError(e)
    
    images = [f for f in os.listdir(
        src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    for image in images:
        filename = os.path.basename(image)
        dest_filename = os.path.join(dest_dir, filename)
        os.rename(os.path.join(src_dir, image), dest_filename)


if __name__ == '__main__':
    # export labels for training subset
    export_labels("../data/balloon/train")
    # export labels for validation subset
    export_labels("../data/balloon/val")
    # relocate images in training directory
    move_images("../data/balloon/train", "../data/balloon/train/images")
    # relocate images in validation directory
    move_images("../data/balloon/val", "../data/balloon/val/images")


import os
import numpy as np
from tqdm import tqdm


def compute_num_classes(lbl_dir_path : str):
    lbls_paths = os.listdir(lbl_dir_path)

    min_class = np.inf
    max_class = -np.inf

    for lbl_path in tqdm(
        iterable=lbls_paths,
        desc="Configuring number of classes",
        total=len(lbls_paths),
        unit=" samples"
    ):
        label = np.loadtxt(os.path.join(lbl_dir_path, lbl_path))

        if label.ndim < 2:
            label = label.reshape(1, -1)

        classes = np.array(label[:, 0]).astype(int)

        if min_class > np.min(classes):
            min_class = np.min(classes)
        if max_class < np.max(classes):
            max_class = np.max(classes)
    
    return min_class, max_class

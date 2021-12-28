
from typing import List, Tuple

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.utils import colorstr


def autoanchors(
    dataloader: DataLoader,
    n_anchors: int = 5,
    outlier_segment: float = 5e-2,
    bandwidth: float = 7e-1,
    outlier_threshold: float = -95e-2,
    verbose: bool = False
) -> Tuple[List[float], List[float]]:
    # parse box areas and register anchors
    anchors, aspects, init_flag = None, None, True
    for _, Y in tqdm(dataloader, desc="Configuring autoanchors", unit=" samples processed"):
        for sample in Y:
            sides = torch.stack([sample['boxes'][:, 2] - sample['boxes']
                                 [:, 0], sample['boxes'][:, 3] - sample['boxes'][:, 1]]).T
            sides = sides.numpy()
            sides = sides[~(sides == 0).any(axis=1)]

            ratios = sides[:, 1] / sides[:, 0]
            ratios = ratios[~(ratios == 0)]

            if init_flag:
                aspects = ratios
                anchors = torch.sqrt(sample['area']).numpy()
                init_flag = False
            else:
                aspects = np.hstack((aspects, ratios))
                anchors = np.hstack(
                    (anchors, torch.sqrt(sample['area']).numpy()))

    anchors = anchors[:, np.newaxis]
    aspects = aspects[:, np.newaxis]

    anchor_raw_n_samples = anchors.shape[0]
    aspect_raw_n_samples = aspects.shape[0]

    anchor_kde = KernelDensity(
        kernel="epanechnikov", bandwidth=bandwidth).fit(anchors)
    anchor_log_dens = anchor_kde.score_samples(anchors)
    anchor_scores = np.exp(anchor_log_dens)

    aspect_kde = KernelDensity(
        kernel="epanechnikov", bandwidth=bandwidth).fit(aspects)
    aspect_log_dens = aspect_kde.score_samples(aspects)
    aspect_scores = np.exp(aspect_log_dens)

    anchor_scores = np.interp(
        anchor_scores, (anchor_scores.min(), anchor_scores.max()), (-1, +1))
    aspect_scores = np.interp(
        aspect_scores, (aspect_scores.min(), aspect_scores.max()), (-1, +1))

    anchor_mask = anchor_scores > outlier_threshold
    aspect_mask = aspect_scores > outlier_threshold

    # remove outliers
    anchors = anchors[anchor_mask]
    aspects = aspects[aspect_mask]

    # sort anchors
    anchors = np.sort(anchors, axis=None)
    aspects = np.sort(aspects, axis=None)

    # configure buckets
    n_anchor_samples = anchors.shape[0]
    anchor_bucket_size = n_anchor_samples // (n_anchors - 1)

    n_aspect_samples = aspects.shape[0]
    aspect_bucket_size = n_aspect_samples // (n_anchors - 1)

    # Removing outliers
    aspects = aspects[:int(n_aspect_samples * (1 - outlier_segment))]
    aspects = aspects[int(n_aspect_samples * outlier_segment):]
    anchors = anchors[:int(n_anchor_samples * (1 - outlier_segment))]
    anchors = anchors[int(n_anchor_samples * outlier_segment):]

    anchor_filtered_n_samples = anchors.shape[0]
    aspect_filtered_n_samples = aspects.shape[0]

    if verbose:
        print(f"\nRemoved {anchor_raw_n_samples - anchor_filtered_n_samples} "
              f"samples out of {anchor_raw_n_samples} from anchor candidates.")
        print(f"Removed {aspect_raw_n_samples - aspect_filtered_n_samples} "
              f"samples out of {aspect_raw_n_samples} from aspect candidates.\n")

    # estimate best anchor sizes
    anchor_sizes = []
    aspect_ratios = []
    aspect_ratios.append(np.around(aspects[0], decimals=2))
    for bucket in range(n_anchors - 1):
        anchor_segment = anchors[anchor_bucket_size *
                                 bucket:anchor_bucket_size * (bucket + 1)]
        aspect_segment = aspects[aspect_bucket_size *
                                 bucket:aspect_bucket_size * (bucket + 1)]

        if bucket < (n_anchors - 1) - 1:
            anchor_sizes.append(np.rint(np.mean(anchor_segment)))
            aspect_ratios.append(
                np.around(np.mean(aspect_segment), decimals=2))
        else:
            # last segment
            anchor_sizes.append(np.rint(np.mean(anchor_segment)))
            anchor_sizes.append(np.rint(anchor_segment[-1]))
            aspect_ratios.append(
                np.around(np.mean(aspect_segment), decimals=2))
            aspect_ratios.append(np.around(aspect_segment[-1], decimals=2))

    aspect_ratios = np.array(aspect_ratios)
    aspect_ratios = np.around(np.linspace(start=np.min(
        aspect_ratios), stop=np.max(aspect_ratios), num=5), decimals=2)

    if max(anchor_sizes) - min(anchor_sizes) < 10.0 or min(anchor_sizes) < 4.0:

        print(f"{colorstr(options=['cyan'], string_args=list(['WARNING']))}: "
              f"You should consider increasing "
              f"{colorstr(options=['red', 'underline'], string_args=list(['img_size']))} "
              f"hyperparameter to gain more "
              f"prediction accuracy.")

    return anchor_sizes, aspect_ratios

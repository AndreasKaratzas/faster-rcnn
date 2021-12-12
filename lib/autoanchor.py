
import torch
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity


def autoanchors(
    dataloader: DataLoader, 
    n_anchors: int = 5, 
    outlier_segment: float = 1e-2, 
    bandwidth: float = 1e-1, 
    outlier_threshold: float = 1e-1
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

    # anchor_universe = np.linspace(np.amin(anchors), np.amax(
    #     anchors), np.prod(anchors.shape) * 10)[:, np.newaxis]
    # aspect_universe = np.linspace(np.amin(aspects), np.amax(
    #     aspects), np.prod(aspects.shape) * 10)[:, np.newaxis]
    
    anchor_kde = KernelDensity(
        kernel="epanechnikov", bandwidth=bandwidth).fit(anchors)
    anchor_log_dens = anchor_kde.score_samples(anchors)
    anchor_mask = (np.exp(anchor_log_dens) < outlier_threshold)

    aspect_kde = KernelDensity(
        kernel="epanechnikov", bandwidth=bandwidth).fit(aspects)
    aspect_log_dens = aspect_kde.score_samples(aspects)
    aspect_mask = (np.exp(aspect_log_dens) < outlier_threshold)

    # remove outliers
    anchors = anchors[anchor_mask]
    aspects = aspects[aspect_mask]

    # sort anchors
    anchors = np.sort(anchors, axis=None)
    aspects = np.sort(aspects, axis=None)

    # configure buckets
    n_samples = anchors.shape[0]
    bucket_size = n_samples // (n_anchors - 1)

    # Removing outliers
    aspects = aspects[:int(n_samples * (1 - outlier_segment))]
    aspects = aspects[int(n_samples * outlier_segment):]
    anchors = anchors[:int(n_samples * (1 - outlier_segment))]
    anchors = anchors[int(n_samples * outlier_segment):]

    # estimate best anchor sizes
    anchor_sizes = []
    aspect_ratios = []
    aspect_ratios.append(np.around(aspects[0], decimals=2))
    for bucket in range(n_anchors - 1):
        anchor_segment = anchors[bucket_size *
                                 bucket:bucket_size * (bucket + 1)]
        aspect_segment = aspects[bucket_size *
                                 bucket:bucket_size * (bucket + 1)]

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

    return anchor_sizes, aspect_ratios

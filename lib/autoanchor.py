
import torch
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader


def autoanchors(dataloader: DataLoader, n_anchors: int = 5, outlier_segment: float = 2e-2) -> Tuple[List[float], List[float]]:
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

            ratios_flipped = sides[:, 0] / sides[:, 1]
            ratios_flipped = ratios_flipped[~(ratios_flipped == 0)]

            if init_flag:
                aspects = np.hstack((ratios, ratios_flipped))
                anchors = torch.sqrt(sample['area']).numpy()
                init_flag = False
            else:
                aspects = np.hstack((aspects, ratios, ratios_flipped))
                anchors = np.hstack(
                    (anchors, torch.sqrt(sample['area']).numpy()))

    # sort anchors
    anchors.sort()
    aspects.sort()

    # configure buckets
    n_samples = anchors.shape[0]

    bucket_size = n_samples // (n_anchors - 1)

    # print(f"\nRemoving outliers ...\n")
    aspects = aspects[:int(n_samples * (1 - outlier_segment))]
    aspects = aspects[int(n_samples * outlier_segment):]
    anchors = anchors[:int(n_samples * (1 - outlier_segment))]
    anchors = anchors[int(n_samples * outlier_segment):]

    # print(f"Minimum aspect ratio found:\t{np.min(aspects)}\nMaximum aspect ratio found:\t{np.max(aspects)}\n")

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

    # output result
    # print(f"Anchor sizes recommended:\t{anchor_sizes}")
    # print(f"Aspect ratios recommended:\t{aspect_ratios}\n")

    return anchor_sizes, aspect_ratios

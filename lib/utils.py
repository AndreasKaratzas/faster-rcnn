
import hashlib
import os
import sys
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from lib.presets import (DetectionPresetEvalTorchVision,
                         DetectionPresetTestTorchVision,
                         DetectionPresetTrainTorchVision)


def colorstr(options, string_args):
    """Usage:
    
    >>> args = ['Andreas', 'Karatzas']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args)} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=list(['Python']))} "
    ...    f"and {colorstr(options=['cyan'], string_args=list(['C++']))}\n")

    Parameters
    ----------
    options : [type]
        [description]
    string_args : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    colors = {'black':          '\033[30m', # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m', # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}
    res = []
    for substr in string_args:
        res.append(''.join(colors[x] for x in options) +
        f'{substr}' + colors['end'])
    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)
    

def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        out = self.model(images, targets)
        return dict_to_tuple(out[0])


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights,
                             global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights,
                         global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
    # Iterate over all model layers
    for layer_number, layer in enumerate(model.modules()):
        # Compute weight histograms for appropriate layer
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            weight_histograms_conv2d(writer, step, weights, layer_number)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, layer_number)


def get_transform(transform_class: str, img_size: int = 640):
    if transform_class == "train":
        return DetectionPresetTrainTorchVision(img_size=img_size)
    elif transform_class == "valid":
        return DetectionPresetEvalTorchVision(img_size=img_size)
    elif transform_class == "test":
        return DetectionPresetTestTorchVision(img_size=img_size)
    else:
        raise ValueError(
            f"Transformation preference was invalid. Value parsed was {transform_class}\n")


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def list2tup(L: List[int]) -> Tuple[Tuple[int]]:
    return tuple([(l,) for l in tuple(L)])


def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from lib.utils import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, f_path: str, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.export_filepath = f_path
        self.log_message = []

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, epoch: int = None):
        i = 1
        if epoch is None:
            raise ValueError(f'Invalid epoch argument ({type(epoch)})')

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            log_dict = {
                'epoch': int(epoch)
            }

            if torch.cuda.is_available():
                log_dict['memory'] = int(round(
                    torch.cuda.memory_reserved() / 1E9))
            else:
                log_dict['memory'] = 0

            for key in self.meters:
                log_dict[key] = round(
                    float(str(self.meters[key]).split()[0]), 3)

            self.log_message.append(log_dict)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        self.time_per_iter = total_time / len(iterable)

    def get_metrics(self):

        self.stats = {}
        for key in self.meters:
            self.stats[key] = round(
                float(str(self.meters[key]).split()[0]), 3)

        return \
            self.stats.get('loss'),           \
            self.stats.get('loss_classifier'),\
            self.stats.get('loss_box_reg'),   \
            self.stats.get('loss_objectness'),\
            self.stats.get('loss_rpn_box_reg')

    def export_data(self):
        with open(self.export_filepath, "a") as f:
            for entry in self.log_message:
                f.write(
                    f"{entry.get('epoch'):8d}{self.time_per_iter:10.2f}"
                    f"{entry.get('lr'):15.3f}{entry.get('loss'):15.3f}"
                    f"{entry.get('loss_classifier'):15.3f}{entry.get('loss_box_reg'):15.3f}"
                    f"{entry.get('loss_objectness'):15.3f}{entry.get('loss_rpn_box_reg'):15.3f}"
                    f"{entry.get('memory'):10d}\n"
                )

        self.log_message = []

import os
import csv
import sys
import argparse
import torch
import numpy as np


class ParseKwargs(argparse.Action):
    """
    From WILDS.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-', '').isnumeric():
                processed_val = int(value_str)

            elif value_str.replace('-', '').replace('.', '').isnumeric():
                processed_val = float(value_str)

            elif value_str in ['True', 'true']:
                processed_val = True

            elif value_str in ['False', 'false']:
                processed_val = False

            else:
                processed_val = value_str

            getattr(namespace, self.dest)[key] = processed_val


def detach(f):
    def wrapper(*args):
        new_args = [args[0]]  # the first arg is the MA object
        for arg in args[1:]:
            if isinstance(arg, torch.Tensor):
                if arg.numel() == 1:
                    arg = arg.item()
                else:
                    arg = arg.cpu().numpy()
            else:
                assert isinstance(arg, (int, float, np.ndarray))
            new_args.append(arg)
        return f(*new_args)
    return wrapper


def safe_divide(a, b):
    if b == 0:
        return -1
    else:
        return a / b


class MABase:
    """
    Base class for Moving Average.
    """
    def __init__(self):
        self.total_loss = 0.
        self.correct = 0
        self.n = 0
        self.fields = ['avg_loss', 'avg_acc']

    def update(self, batch_loss, pred, label):
        batch_size = len(label)
        self.total_loss += batch_size * batch_loss
        self.correct += np.sum(pred == label)
        self.n += batch_size

    @property
    def avg_loss(self):
        return safe_divide(self.total_loss, self.n)

    @property
    def avg_acc(self):
        return safe_divide(self.correct, self.n)

    def tostring(self):
        s = ''
        for f in self.fields:
            s += '{:s}: {:.4f}\n'.format(f, getattr(self, f))
        return s

    def todict(self, prefix=''):
        return {prefix+f: getattr(self, f) for f in self.fields}

    def reset(self):
        self.__init__()


class MABinary(MABase):
    """
    Moving Average in binary classification.
    """
    def __init__(self):
        super().__init__()
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.fields.extend(['recall', 'precision', 'f1'])

    def update(self, batch_loss, pred, label):
        super().update(batch_loss, pred, label)
        self.tp += np.sum((pred == label) & (pred == 1))
        self.tn += np.sum((pred == label) & (pred == 0))
        self.fp += np.sum((pred != label) & (pred == 1))
        self.fn += np.sum((pred != label) & (pred == 0))

    @property
    def recall(self):
        return safe_divide(self.tp, self.tp + self.fn)

    @property
    def precision(self):
        return safe_divide(self.tp, self.tp + self.fp)

    @property
    def f1(self):
        return safe_divide(2 * self.recall * self.precision,
                           self.recall + self.precision)


class Logger:
    def __init__(self, dest, mode='w', stdout=False):
        self.console = sys.stdout if stdout else None
        self.file = open(dest, mode)

    def __del__(self):
        self.close()

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.console is not None:
            self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        if self.console is not None:
            self.console.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.console is not None:
            self.console.close()
        self.file.close()


class MultiDomCSVLogger:
    def __init__(self, dest, domain2id, mode='w', binary=True):
        self.file = open(dest, mode)
        self.binary = binary
        self.ma = MABinary() if binary else MABase()
        self.group_ma = {}  # domain id -> domain moving average
        self.id2domain = {v: k for k, v in domain2id.items()}

    def get_overall_ma(self):
        return self.ma.tostring()

    @detach
    def update(self, batch_loss, pred, label, domain_ids):
        """
        Update overall loss and acc. Also update the domain acc
        for those domains appearing in domain_ids.
        """
        self.ma.update(batch_loss, pred, label)
        unique = np.unique(domain_ids)
        for domain in unique:
            if domain not in self.group_ma:
                self.group_ma[domain] = MABinary() if self.binary else MABase()
            mask = domain_ids == domain
            self.group_ma[domain].update(-1, pred[mask], label[mask])

    def write(self):
        """
        Write the overall loss and acc, as well as the domainwise acc into csv.
        This should be performed exactly once an epoch.
        """
        row = self.ma.todict()
        fields = self.ma.fields
        for domain_id in self.group_ma:
            domain_name = self.id2domain[domain_id]
            local = self.group_ma[domain_id].todict(prefix=f'{domain_name}_')
            fields = fields + local.keys()
            row.update(local)
        writer = csv.DictWriter(self.file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        self.reset()

    def reset(self):
        self.ma.reset()
        for k in self.group_ma:
            self.group_ma[k].reset()

    def close(self):
        self.file.close()

import os
import csv
import sys
import random
import argparse
import torch
import numpy as np
from datetime import datetime


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


def set_seed(seed):
    """
    Make the results reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pretty_args(args):
    """
    Return a nice string repr of the argparse.Namespace object.
    """
    s = 'Args:\n'
    for name, val in vars(args).items():
        s += f'{name.replace("_", " ").capitalize()}: {val}\n'
    return s + '\n'


def detach(f):
    """
    Decorator that migrates the arguments of function f to ndarray or scalar.
    """
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


def save_with_aux(module, epoch, best_metric, path):
    """
    Save the state dict of module as well as some auxiliary info.
    """
    state = {}
    state['module'] = module.state_dict()
    state['epoch'] = epoch
    state['best_metric'] = best_metric
    torch.save(state, path)


def load_with_aux(module, path):
    """
    The inverse of `save_with_aux`.
    """
    state = torch.load(path)
    module.load_state_dict(state['module'])
    return state['epoch'], state['best_metric']


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
        batch_size = 1 if isinstance(label, int) else len(label)
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
    def __init__(self, dest, mode='w'):
        self.console = sys.stdout
        self.file = open(dest, mode)

    def __del__(self):
        self.close()

    def __exit__(self, *args):
        self.close()

    def write(self, msg, stdout=True, time=True):
        msg += '\n'
        if time:
            now = datetime.now()
            time_s = now.strftime('%d/%m %H:%M:%S')
            msg = time_s + ' - ' + msg

        if stdout:
            self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        self.file.close()


class MultiDomCSVLogger:
    def __init__(self, dest, domain2id, mode='w',
                 binary=True, metric='avg_acc'):

        self.file = open(dest, mode)
        self.binary = binary
        self.metric = metric

        self.ma = MABinary() if binary else MABase()
        assert hasattr(self.ma, self.metric)

        self.group_ma = {}  # domain id -> domain moving average
        self.id2domain = {v: k for k, v in domain2id.items()}

        self.has_setup = False
        self.writer = None

    def get_overall_ma(self):
        return self.ma.tostring()

    def get_metric(self):
        return getattr(self.ma, self.metric)

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

    def setup(self, fields):
        """
        Setup the csv. Write the head.
        """
        assert (not self.has_setup) and (self.writer is None)
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.has_setup = True

    def write(self, epoch):
        """
        Write the overall loss and acc, as well as the domainwise acc into csv.
        This should be performed exactly once an epoch.
        """
        row = {'Epoch': epoch}
        row.update(self.ma.todict())
        for domain_id in self.group_ma:
            domain_name = self.id2domain[domain_id]
            local = self.group_ma[domain_id].todict(prefix=f'{domain_name}_')
            row.update(local)
        if not self.has_setup:
            self.setup(list(row.keys()))
        self.writer.writerow(row)

    def reset(self):
        self.ma.reset()
        for k in self.group_ma:
            self.group_ma[k].reset()

    def close(self):
        self.file.close()

import os
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from model import init_model
from model.models import BertClassifier
from train import parse_args
from algo import init_algorithm
from utils import MABinary, safe_divide

from typing import List, Union, Dict, Tuple


def parse_log_to_args(fi, names=None):
    """
    Parse a log.txt file to a dict that contains corresponding args.
    NOTE: The values are all in string representations.

    Parameters:
        fi: input stream
        names: the names of the needed args
    """
    if names is None:
        names = ['algorithm', 'target', 'lr', 'wd', 'lr2', 'wd2',
                 'alpha_d', 'alpha_meta', 'hidden_size_d', 'num_hidden_d']

    args = {}
    for i, line in enumerate(fi):
        if i == 0:
            continue
        if line.startswith('N train steps'):
            break

        name, value = line.strip().split(': ', maxsplit=1)
        name = name.strip().lower().replace(' ', '_')
        value = value.strip()

        if name not in names or value.lower() == 'none':
            continue
        args[name] = value
    return args


def combine_all_csv(root, arg_names, extra_fields=[]):
    """
    Combine all csv tables under `root` in a large one, with the args as index.
    """

    splits = ['train', 'valid', 'test']
    fields = ['avg_loss', 'avg_acc', 'f1']
    fields.extend(extra_fields)
    full = []

    for dirpath, _, filenames in os.walk(root):
        if 'log.txt' not in filenames:
            continue
        with open(os.path.join(dirpath, 'log.txt'), 'r') as fi:
            args = parse_log_to_args(fi, arg_names)

        tables = []
        has_results = True

        for split in splits:
            try:
                table = pd.read_csv(os.path.join(
                    dirpath, f'{split}_epochs.csv'))
            except pd.errors.EmptyDataError:
                print(f'Cannot read from {split} in {dirpath}')
                has_results = False
                break

            table = table.set_index('Epoch')
            table = table[fields]  # only avg_loss, avg_acc, etc

            # add split information
            table.columns = pd.MultiIndex.from_tuples(
                zip([split] * len(fields), table.columns)
            )
            tables.append(table)

        # if error occurs
        if not has_results:
            continue

        # table corresponds to one job
        table = pd.concat(tables, axis=1)

        # add algorithm, target and hparams information
        table['algo'] = args['algorithm']
        table['tgt'] = args['target']
        table['hparams'] = ', '.join(f'{k}={v}' for k, v in args.items()
                                     if k not in ['algorithm', 'target'])
        table = table.reset_index()  # recover Epoch
        table = table.set_index(['tgt', 'algo', 'hparams', 'Epoch'])

        full.append(table)

    return pd.concat(full).sort_index()


def load_bert_classifier(path):
    prefix = 'model.'  # the prefix to remove

    state_dict = torch.load(path)['module']
    state_dict = {k[len(prefix):]: v for k, v in state_dict.items()
                  if k.startswith(prefix)}

    bc = BertClassifier(2, 'cuda:0', need_hook=True)
    bc.load_state_dict(state_dict)
    return bc


def get_model_from_root(root, prefix):
    """
    Get the model that matches the given prefix. One has to make sure `prefix`
    uniquely specifies one model in the root.
    """
    m = None
    for d in os.listdir(root):
        full_d = os.path.join(root, d)
        if os.path.isdir(full_d) and d.startswith(prefix):
            m = load_bert_classifier(os.path.join(full_d, 'best_model.pth'))
            break
    assert m is not None, 'Model not found!'
    return m


def get_algorithm_from_root(root, prefix):
    """
    Get the whole algorithm object that matches the given prefix.
    """
    for d in os.listdir(root):
        full_d = os.path.join(root, d)
        if os.path.isdir(full_d) and d.startswith(prefix):
            with open(os.path.join(full_d, 'log.txt'), 'r') as fi:
                args_d = parse_log_to_args(fi)
            args_s = ' '.join([f'--{k} {v}' for k, v in args_d.items()])
            args = parse_args(cmd_line=args_s)
            args.n_train_steps = 0

            model = init_model(args.model, 'cuda:0', 2, args)
            algorithm = init_algorithm(args.algorithm, 'cuda:0', model,
                                       None, args)

            algorithm.load_state_dict(torch.load(os.path.join(
                full_d, 'best_model.pth'))['module'])

            return algorithm.to('cuda:0')


def do_for_all_algo_and_tgt(algo: List[str], tgt: Union[str, List[str]]):
    """Return a decorator. This decorator would run the decorated function
    with all possible (algorithm, target) pairs as arguments.
    The decorated function must accept `algorithm` and `target` as keyword
    arguments in declaration.
    """
    def decorator(f):
        def g(*args, **kwargs):
            for a in algo:
                for t in tgt:
                    f(*args, algorithm=a, target=t, **kwargs)
        return g
    return decorator


def display_metrics_nicely(metrics: Dict[str, float], header: bool = True):
    """Display the metrics like accuracy, precision, etc., in a nice view.
    """
    cand = [('N', '{:<8s}', '{:<8d}'),
            ('Acc', '{:<8s}', '{:<8.3f}'),
            ('Precision', '{:<12s}', '{:<12.3f}'),
            ('Recall', '{:<8s}', '{:<8.3f}'),
            ('F1', '{:<8s}', '{:<8.3f}'),
            ('Specificity', '{:<16s}', '{:<16.3f}')]
    if header:
        dash = '-' * 64
        print(dash)
        print(''.join([b.format(a) for a, b, _ in cand if a in metrics]))
        print(dash)
    print(''.join([c.format(metrics[a]) for a, _, c in cand if a in metrics]))


def cmp_binary_pred_and_label(pred: List[int],
                              label: List[int]) -> Dict[str, float]:
    """Compare the predictions and labels in a binary setting, assuming they
    are either 0 or 1. Calculate acc, precision, recall, f1 and specificity.
    """
    assert len(pred) != 0 and len(pred) == len(label)
    stats = MABinary()
    stats.update(0, np.array(pred), np.array(label))
    spec = safe_divide(stats.tn, stats.tn + stats.fp)

    # keys and values
    keys = ['N', 'Acc', 'Precision', 'Recall', 'F1', 'Specificity']
    vals = [stats.n, stats.avg_acc, stats.precision,
            stats.recall, stats.f1, spec]

    return dict(zip(keys, vals))


@torch.no_grad()
def get_pred_and_prob(model,
                      all_txt: List[str],
                      batch_size: int = 64) -> Tuple[List[str], List[float]]:
    """Get the predictions and prediction probabilities using model on
    the given list of texts. The model should be a BertClassifier.
    """
    all_pred = []
    all_pred_prob = []
    loader = DataLoader(all_txt, batch_size=batch_size, drop_last=False)

    for batch in tqdm(loader):
        assert isinstance(batch, list)
        assert isinstance(batch[0], str)

        logits = model(batch)
        pred = torch.max(logits, dim=1)[1]
        prob = softmax(logits, dim=1)[range(len(pred)), pred]

        all_pred.extend(['positive' if x == 1 else 'negative'
                         for x in pred.tolist()])
        all_pred_prob.extend(prob.tolist())

    assert len(all_pred) == len(all_txt) and len(all_pred_prob) == len(all_txt)
    return all_pred, all_pred_prob

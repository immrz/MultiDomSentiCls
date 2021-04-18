import os
import torch
import pandas as pd

from model.models import BertClassifier
from train import parse_args
from algo import init_algorithm
from model import init_model


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
    for d in os.listdir(root):
        full_d = os.path.join(root, d)
        if os.path.isdir(full_d) and d.startswith(prefix):
            return load_bert_classifier(os.path.join(full_d, 'best_model.pth'))


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

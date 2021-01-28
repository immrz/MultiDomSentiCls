from data.datasets import ATMFDataset
from model import init_model
from algo import init_algorithm
from utils import ParseKwargs

from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser()

    # required arg
    parser.add_argument('--algorithm', type=str, required=True)

    # IO arg
    parser.add_argument('--path', type=str, default='data/preprocessed.csv')
    parser.add_argument('--log_dir', type=str, default='log/')

    # algorithm arg
    # TODO

    # model arg
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')

    # dataloader arg
    parser.add_argument('--batch_size', type=int)

    # optimization arg
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*',
                        action=ParseKwargs, default={})
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--scheduler_kwargs', nargs='*',
                        action=ParseKwargs, default={})

    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume', action='store')

    args = parser.parse_args()
    # TODO: initialize empty fields
    return args


def run_epoch():
    pass


def main(args):
    pass

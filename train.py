from model import init_model
from algo import init_algorithm
from data import init_dataset
from utils import ParseKwargs, set_seed, Logger, pretty_args, save_with_aux
from hparams_registry import populate_args

from argparse import ArgumentParser
import os
from tqdm.auto import tqdm

import torch


def parse_args(cmd_line=None):
    parser = ArgumentParser()

    # required arg
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)

    # IO arg
    parser.add_argument('--dataset', type=str, default='ATMF')
    parser.add_argument('--path', type=str)
    parser.add_argument('--log_dir', type=str,
                        default=os.environ.get('PT_OUTPUT_DIR', 'log/'))

    # algorithm arg
    # DANN
    parser.add_argument('--alpha_d', type=float)
    parser.add_argument('--hidden_size_d', type=int)
    parser.add_argument('--num_hidden_d', type=int)
    parser.add_argument('--lr2', type=float)
    parser.add_argument('--wd2', type=float)
    parser.add_argument('--extra_losses', type=str, nargs='*')
    parser.add_argument('--n_domains', type=int,
                        help='output size of discriminator')

    # MLDG
    parser.add_argument('--alpha_meta', type=float)

    # model arg
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_token_len', type=int)

    # dataloader arg
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=16)

    # optimization arg
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--valid_metric', type=str, default='avg_acc')
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
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--save_last', action='store_true')

    # parse args
    if cmd_line is None:
        args = parser.parse_args()
    else:
        # efficient for ipython
        args = parser.parse_args(cmd_line.split())

    # initialize empty fields
    # don't forget to add n_train_steps after getting training set
    args = populate_args(args)

    return args


def run_epoch(algorithm, dataset, logger, args, epoch, train=False):
    loader = dataset['loader']
    csv_logger = dataset['logger']
    iterator = loader if args.verbose else tqdm(loader)

    for step, batch in enumerate(iterator):
        x, y, domain_ids = batch

        if train:
            algorithm.train()
            batch_results = algorithm.update(batch)

        else:
            algorithm.eval()
            batch_results = algorithm.predict(x, y=y)

        loss = batch_results.loss
        pred = torch.max(batch_results.logits, 1)[1]

        # update moving average of loss and acc
        csv_logger.update(loss, pred, y, domain_ids)

        # if meet logging criterion
        if train and (step + 1) % args.log_every == 0:
            current_ma = csv_logger.get_overall_ma()
            logger.write(current_ma + '\n', stdout=args.verbose)

    # write the epoch results
    metric = csv_logger.get_metric()
    result = csv_logger.get_overall_ma()
    logger.write(result + '\n')
    csv_logger.write(epoch)
    csv_logger.reset()

    return metric


def main(args):
    set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # build dataset
    full_dataset, datasets = init_dataset(args.dataset, args)

    # create global logger
    logger = Logger(os.path.join(args.log_dir, 'log.txt'))

    # log args and datasets
    logger.write(pretty_args(args))
    for split in datasets:
        logger.write(f'{split.capitalize()} Split:')
        logger.write(full_dataset.pretty_stats(split=split), time=False)

    if args.dry_run:
        return

    # build model and algorithm
    model = init_model(args.model, args.device, full_dataset.n_classes, args)
    algorithm = init_algorithm(args.algorithm, args.device, model, args)

    # model selection
    best_metric = -1e6
    best_save = os.path.join(args.log_dir, 'best_model.pth')

    # training starts
    for epoch in range(args.n_epochs):
        logger.write(f'Epoch: {epoch}')

        # training
        logger.write('Training...')
        run_epoch(algorithm, datasets['train'],
                  logger, args, epoch, train=True)

        # validation
        logger.write('Validation...')
        with torch.no_grad():
            metric = run_epoch(algorithm, datasets['valid'],
                               logger, args, epoch)
        if metric > best_metric:
            best_metric = metric
            if args.save_best:
                logger.write(f'Saving best at epoch {epoch}')
                save_with_aux(algorithm, epoch, best_metric, best_save)

        # other splits
        for split in [s for s in datasets if s not in ['train', 'valid']]:
            logger.write(f'{split.capitalize()} Split:')
            with torch.no_grad():
                run_epoch(algorithm, datasets[split],
                          logger, args, epoch)

    # finished
    if args.save_last:
        save_with_aux(algorithm, args.n_epochs - 1, best_metric,
                      os.path.join(args.log_dir, 'last_model.pth'))
    logger.write('Finished')
    logger.close()
    for split in datasets:
        datasets[split]['logger'].close()


if __name__ == '__main__':
    args = parse_args()
    main(args)

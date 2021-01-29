import numpy as np


def _hparams(algorithm, dataset, seed):
    """
    From DomainBed.
    """
    hparams = {}

    def register(name, default, random_fn):
        """
        Each hparam is a tuple of (default value, random value).
        """
        assert name not in hparams
        random_state = np.random.RandomState(seed)
        hparams[name] = (default, random_fn(random_state))

    register('model', 'bert-base-uncased', lambda r: 'bert-base-uncased')
    register('batch_size', 16, lambda r: 2**r.randint(2, 5))
    register('n_epochs', 3, lambda r: r.choice([3, 4]))
    register('optimizer', 'AdamW', lambda r: 'AdamW')
    register('lr', 2e-5, lambda r: r.randint(2, 6)*10**r.randint(-6, -4))
    register('wd', 0.01, lambda r: 10**r.randint(-4, 0))
    register('scheduler', 'linear_schedule_with_warmup',
             lambda r: 'linear_schedule_with_warmup')

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}


def populate_args(args, random=False, overwrite=False):
    d = args.__dict__
    p = random_hparams(args.algorithm, args.dataset, args.seed) \
        if random else default_hparams(args.algorithm, args.dataset)

    for k, v in p.items():
        if (k not in d) or (d[k] is None) or overwrite:
            d[k] = v

    # complete the args
    if args.scheduler == 'linear_schedule_with_warmup':
        args.scheduler_kwargs = {'num_warmup_steps': 0}
    if args.dataset == 'ATMF':
        args.path = 'data/preprocessed.csv'

    return args

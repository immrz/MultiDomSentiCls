import numpy as np
import os


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

    # common hparams
    register('max_token_len', 512, lambda r: r.choice([128, 256, 512]))
    register('batch_size', 8, lambda r: r.choice([4, 8, 16, 32]))
    register('n_epochs', 3, lambda r: r.choice([3, 4]))
    register('lr', 2e-5, lambda r: r.randint(2, 6)*10**r.randint(-6, -4))
    register('wd', 0.01, lambda r: 10**r.randint(-4, 0))
    hparams['optimizer'] = ('AdamW',)
    hparams['scheduler'] = ('linear_schedule_with_warmup',)
    hparams['extra_losses'] = ([],)

    if algorithm in ['DANN', 'CDAN']:  # DANN hparams
        register('lr2', 2e-5, lambda r: 10**r.uniform(-5, -3.5))
        register('wd2', 0., lambda r: 10**r.uniform(-6, -2))
        register('alpha_d', 0.1, lambda r: 10**r.uniform(-3, 1))
        register('hidden_size_d', 1024, lambda r: r.choice([256, 512, 1024]))
        register('num_hidden_d', 1, lambda r: r.randint(0, 4))
        register('n_iter_d', 0, lambda r: r.randint(1, 11))
        hparams['extra_losses'] = (['disc_loss'],)
    elif algorithm == 'MLDG':
        register('lr2', 2e-5, lambda r: 10**r.uniform(-5, -3.5))
        register('wd2', 0., lambda r: 10**r.uniform(-6, -2))
        register('alpha_meta', 1.0, lambda r: 10**r.uniform(-3, 1))
        hparams['extra_losses'] = (['meta_loss'],)
    elif algorithm == 'IRM':
        register('penalty_weight', 100., lambda r: 10.**r.uniform(2, 5))
        register('penalty_anneal_iters', 500,
                 lambda r: r.choice([100, 500, 1000]))
    elif algorithm != 'ERM':
        raise NotImplementedError

    if dataset == 'ATMF':
        hparams['n_domains'] = (3,)
    else:
        raise NotImplementedError

    return hparams


def default_hparams(algorithm, dataset):
    return {k: v[0] for k, v in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {k: v[-1] for k, v in _hparams(algorithm, dataset, seed).items()}


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
        args.path = os.path.join(
            os.environ.get('PT_DATA_DIR', 'data/data_root'),
            'preprocessed.csv',
        )

    return args

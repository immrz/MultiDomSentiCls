from torch.utils.data import DataLoader
from collections import defaultdict
from utils import MultiDomCSVLogger
from data.datasets import ATMFDataset
import os


def init_dataset(name, args):
    datasets = defaultdict(dict)

    if name == 'ATMF':
        full_dataset = ATMFDataset(args.path, args.target)
    else:
        raise NotImplementedError

    for split in full_dataset.split_dict:
        data = full_dataset.get_subset(split)
        loader_kwargs = {'dataset': data,
                         'num_workers': args.num_workers,
                         'batch_size': args.batch_size,
                         'pin_memory': True}

        if split == 'train':
            loader = DataLoader(shuffle=True, drop_last=True, **loader_kwargs)
            args.n_train_steps = len(loader) * args.n_epochs

        else:
            loader = DataLoader(shuffle=False, **loader_kwargs)

        datasets[split]['data'] = data
        datasets[split]['loader'] = loader
        datasets[split]['logger'] = MultiDomCSVLogger(
            os.path.join(args.log_dir, f'{split}_epochs.csv'),
            full_dataset.domain2id,
            binary=full_dataset.n_classes == 2,
            metric=args.valid_metric,
            extra_losses=args.extra_losses,
        )

    return full_dataset, datasets

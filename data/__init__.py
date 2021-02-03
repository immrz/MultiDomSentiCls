import numpy as np
import torch
from torch.utils.data import DataLoader

import os
from collections import defaultdict

from utils import MultiDomCSVLogger
from data.datasets import ATMFDataset, ATMFOracleDataset


def init_dataset(name, args):
    datasets = defaultdict(dict)

    if name == 'ATMF':
        if args.oracle:
            assert args.algorithm == 'ERM'
            full_dataset = ATMFOracleDataset(args.path, args.target)
        else:
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
            if args.algorithm == 'MLDG':
                # sample from all source domains each time
                bs = DomainSampler(full_dataset.domain_ids[data.indices],
                                   n_samples_per_domain=args.batch_size)
                loader = DataLoader(data,
                                    batch_sampler=bs,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
            else:
                loader = DataLoader(shuffle=True, drop_last=True,
                                    **loader_kwargs)
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


class DomainSampler:
    """
    Batch sampler that samples `n_samples_per_domain` instances from each
    domain that appears in `domain_ids` and constructs them into a batch.

    If `align` is set to `padding`, short domains are padded to have the
    same length as the longest domain.
    If `align` is set to `truncation`, long domains are truncated to the
    length of the shortest domain.

    The last batch is dropped if it's incomplete.
    """
    def __init__(self,
                 domain_ids: torch.Tensor,
                 n_samples_per_domain: int,
                 align='padding'):

        assert align in ['padding', 'truncation']
        self.n_samples_per_domain = n_samples_per_domain
        self.unique_ids = torch.unique(domain_ids)  # unique domain ids

        self.indices = []  # indices of elements of each domain id
        for d in self.unique_ids:
            d_indices = torch.nonzero(domain_ids == d, as_tuple=True)[0]
            self.indices.append(d_indices)

        # specify the length of the dataset
        max_len = max([len(x) for x in self.indices])
        min_len = min([len(x) for x in self.indices])
        self.n_samples = max_len if align == 'padding' else min_len

    def __iter__(self):
        indices = []

        for d_indices in self.indices:
            d_indices = d_indices.detach().clone()
            idx_len = len(d_indices)
            if idx_len < self.n_samples:
                pad = d_indices[np.random.choice(
                    idx_len, self.n_samples - idx_len)]
                d_indices = torch.cat((d_indices, pad))
            elif idx_len > self.n_samples:
                d_indices = d_indices[np.random.choice(
                    idx_len, self.n_samples, replace=False)]

            # shuffle d_indices
            d_indices = d_indices[torch.randperm(len(d_indices))]
            indices.append(d_indices)

        for i in range(len(self)):
            a = i * self.n_samples_per_domain
            b = a + self.n_samples_per_domain
            batch_idx = torch.cat([d_indices[a:b] for d_indices in indices])
            yield batch_idx.tolist()

    def __len__(self):
        return self.n_samples // self.n_samples_per_domain

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Subset


class MultiDomainDataset:
    def __init__(self):
        self.check_init()

    def check_init(self):
        required_attrs = ['_y_array', '_domain_ids',
                          '_split_array', '_split_dict',
                          '_n_classes', '_domain2id']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'Missing attr {attr_name}'

        assert len(self._y_array) == len(self._domain_ids)
        assert len(self._y_array) == len(self._split_array)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        domain_id = self.domain_ids[idx]
        return x, y, domain_id

    def get_subset(self, split):
        assert split in self.split_dict, f'Split {split} not found'
        mask = self.split_array == self.split_dict[split]
        split_idx = np.where(mask)[0]
        return Subset(self, split_idx)

    def init_domain2id(self, domains, target):
        self._domain2id = {}
        for i, d in enumerate(x for x in domains if x != target):
            self._domain2id[d] = i
        self._domain2id[target] = len(domains) - 1  # target always the last

    @property
    def y_array(self):
        return self._y_array

    @property
    def split_array(self):
        """
        np.ndarray. Same length as the dataset. Contains the split id
        of each sample.
        """
        return self._split_array

    @property
    def domain_ids(self):
        return self._domain_ids

    @property
    def split_dict(self):
        """
        str -> int. Map keys like 'train' and 'valid' to ids.
        """
        return self._split_dict

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def domain2id(self):
        return self._domain2id

    def get_input(self, idx):
        """
        Get x at idx. Should be overloaded because different datasets
        may have different ways to store x.
        """
        raise NotImplementedError

    def init_split(self, target):
        raise NotImplementedError

    def pretty_stats(self, split=None):
        """
        Return a pretty string representation of the statistics of the certain
        split of this dataset. If split is None, use the full dataset instead.
        """
        raise NotImplementedError


class ATMFDataset(MultiDomainDataset):
    def __init__(self, path_to_csv, target):
        self.read_data(path_to_csv, target)
        self.texts = self.data['text']

        self._n_classes = 2
        self.label2id = {'positive': 1, 'negative': 0}
        self._y_array = torch.tensor(
            self.data['label'].replace(self.label2id).values).long()

        self.init_domain2id(['A', 'T', 'M', 'F'], target)
        self._domain_ids = torch.tensor(
            self.data['domain'].replace(self._domain2id).values).long()

        self.init_split(target)
        super(ATMFDataset, self).__init__()

    def read_data(self, path_to_csv, target):
        self.data = pd.read_csv(path_to_csv)

    def get_input(self, idx):
        return self.texts.iloc[idx]

    def init_split(self, target):
        """
        The part whose domains match target will be used as test split.
        The remaining part will be split into train and valid according
        to `valid` column in the DataFrame.
        """
        self._split_dict = {'train': 0, 'valid': 1, 'test': 2}
        split_array = np.zeros(len(self.data))

        test_mask = (self.data['domain'] == target).values
        split_array[test_mask] = self._split_dict['test']

        valid_mask = self.data['valid'].values
        split_array[valid_mask & (~test_mask)] = self._split_dict['valid']

        self._split_array = split_array

    def pretty_stats(self, split=None):
        gb = ['domain', 'label']
        if split is None:
            info = self.data.groupby(gb).size()
        else:
            mask = self.split_array == self.split_dict[split]
            sub_data = self.data.loc[mask]
            info = sub_data.groupby(gb).size()
        return repr(info) + '\n'


class ATMFOracleDataset(ATMFDataset):
    def __init__(self, path_to_csv, target, ratio=(0.6, 0.2, 0.2)):
        self.ratio = ratio
        assert sum(ratio) == 1
        super().__init__(path_to_csv, target)

    def read_data(self, path_to_csv, target):
        data = pd.read_csv(path_to_csv)
        data = data[data['domain'] == target]
        self.data = data

    def init_domain2id(self, domains, target):
        self._domain2id = {target: 0}  # only use target domain

    def init_split(self, target):
        n = len(self._y_array)
        self._split_dict = {'train': 0, 'valid': 1, 'test': 2}
        split_array = np.zeros(n)

        perm = np.random.permutation(n)
        num_tr, num_va = int(n * self.ratio[0]), int(n * self.ratio[1])
        split_array[perm[num_tr:num_tr+num_va]] = self._split_dict['valid']
        split_array[perm[num_tr+num_va:]] = self._split_dict['test']

        self._split_array = split_array

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
                          '_n_classes']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'Missing attr {attr_name}'

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

    @property
    def y_array(self):
        return self._y_array

    @property
    def split_array(self):
        return self._split_array

    @property
    def domain_ids(self):
        return self._domain_ids

    @property
    def split_dict(self):
        return self._split_dict

    @property
    def n_classes(self):
        return self._n_classes

    def get_input(self, idx):
        raise NotImplementedError

    def init_split(self, target):
        raise NotImplementedError


class ATMFDataset(MultiDomainDataset):
    def __init__(self, path_to_csv, target):
        self.data = pd.read_csv(path_to_csv)

        self._n_classes = 2
        self.label2id = {'positive': 1, 'negative': 0}
        self._y_array = torch.tensor(
            self.data['label'].replace(self.label2id).values).long()

        self.domain2id = {'A': 0, 'T': 1, 'M': 2, 'F': 3}
        self._domain_ids = torch.tensor(
            self.data['domain'].replace(self.domain2id).values).long()

        self.init_split(target)
        super(ATMFDataset, self).__init__()

    def get_input(self, idx):
        return self.data.loc[idx, 'text']

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

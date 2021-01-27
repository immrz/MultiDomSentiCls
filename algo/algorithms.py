import torch.nn as nn


class Algorithm(nn.Module):
    def __init__(self, model, optimizer, scheduler, hparams):
        super(Algorithm, self).__init__()
        self.model = self.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hparams = hparams

    def update(self, batch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    def update(self, batch):


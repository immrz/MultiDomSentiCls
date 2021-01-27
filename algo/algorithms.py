from utils import AlgOut
import torch.nn as nn
import torch.nn.functional as F


class Algorithm(nn.Module):
    def __init__(self, model, optimizer, scheduler, device, hparams):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.hparams = hparams

    def update(self, batch):
        raise NotImplementedError

    def predict(self, x, y=None):
        raise NotImplementedError


class ERM(Algorithm):
    def update(self, batch):
        self.optimizer.zero_grad()
        x, y, _ = batch
        out = self.predict(x, y=y)
        out.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return out

    def predict(self, x, y=None):
        logits = self.model(x)
        loss = None if not y else F.cross_entropy(logits, y.to(self.device))
        return AlgOut(loss=loss, logits=logits)

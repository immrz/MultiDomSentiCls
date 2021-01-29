import torch.nn as nn
import torch.nn.functional as F

from algo.utils import AlgOut, get_optimizer, get_scheduler


class Algorithm(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = get_optimizer(args.optimizer, model, args)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer,
                                       args.n_train_steps, args)
        self.device = device
        self.args = args

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
        if self.scheduler is not None:
            self.scheduler.step()
        return out

    def predict(self, x, y=None):
        logits = self.model(x)
        loss = None if y is None else \
            F.cross_entropy(logits, y.to(self.device))
        return AlgOut(loss=loss, logits=logits)

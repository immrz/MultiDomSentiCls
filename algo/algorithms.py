import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.utils import AlgOut, get_optimizer, get_scheduler
from model.models import GRL, MLP


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


class DANN(ERM):
    def __init__(self, model, device, n_domains, num_hidden_d,
                 hidden_size_d, alpha_d, args):
        super().__init__(model, device, args)
        self.alpha_d = alpha_d

        # discriminator
        self.disc = MLP(self.model.out_size, n_domains,
                        num_hidden_d, hidden_size_d)

        # optimizer of discriminator
        self.opt_disc = optim.Adam(self.disc.parameters(),
                                   lr=args.lr2,
                                   weight_decay=args.wd2)

    def update(self, batch):
        self.optimizer.zero_grad()
        self.opt_disc.zero_grad()

        x, y, domain_ids = batch
        out = self.predict(x, y=y)
        emb = self.model.emb  # feature embeddings

        # adversarial training
        disc_logits = self.disc(GRL.apply(emb))
        disc_loss = F.cross_entropy(disc_logits, domain_ids)

        # backward
        total_loss = out.loss + self.alpha_d * disc_loss
        total_loss.backward()
        self.optimizer.step()
        self.opt_disc.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # also return discriminator loss
        out.loss = {'loss': out.loss, 'disc_loss': disc_loss}

        return out

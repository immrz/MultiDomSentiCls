import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.utils import AlgOut, get_optimizer, get_scheduler, random_pair_batch
from model.models import GRL, MLP

import copy


class Algorithm(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = get_optimizer(args.optimizer, model, args)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args)
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
    def __init__(self,
                 model,
                 device,
                 n_domains,  # number of domains for training
                 num_hidden_d,
                 hidden_size_d,
                 alpha_d,
                 n_iter_disc,  # number of iterations of training discriminator
                 train_loader_disc,  # data loader for discriminator
                 args):

        super().__init__(model, device, args)
        self.alpha_d = alpha_d

        self.n_iter_disc = n_iter_disc
        self.train_loader_disc = train_loader_disc
        self.train_iter_disc = iter(train_loader_disc)

        # whether to force disc output to be uniform in the outer optimization
        self.force_uniform = args.force_uniform

        # discriminator
        self.disc = MLP(self.model.out_size, n_domains,
                        num_hidden_d, hidden_size_d)

        # optimizer of discriminator
        self.opt_disc = optim.Adam(self.disc.parameters(),
                                   lr=args.lr2,
                                   weight_decay=args.wd2)

    def _discriminate(self, in_emb, domain_ids):
        disc_logits = self.disc(in_emb)
        disc_loss = F.cross_entropy(disc_logits, domain_ids.to(self.device))
        return disc_logits, disc_loss

    def update(self, batch):
        # train the discriminator alone first
        for _ in range(self.n_iter_disc):
            self.opt_disc.zero_grad()

            # get unlabeled data
            try:
                unl_batch = next(self.train_iter_disc)
            except StopIteration:
                self.train_iter_disc = iter(self.train_loader_disc)
                unl_batch = next(self.train_iter_disc)

            # feed inputs and get embeddings
            with torch.no_grad():
                x, _, domain_ids = unl_batch
                _ = self.predict(x)
                emb = self.model.emb

            # max discriminator acc
            _, disc_loss = self._discriminate(emb, domain_ids)
            disc_loss.backward()
            self.opt_disc.step()

        # jointly train classifier, featurizer and discriminator
        self.optimizer.zero_grad()
        self.opt_disc.zero_grad()

        x, y, domain_ids = batch
        out = self.predict(x, y=y)
        emb = self.model.emb  # feature embeddings

        if not self.force_uniform:
            # adversarial training
            _, disc_loss = self._discriminate(GRL.apply(emb), domain_ids)

            # backward
            total_loss = out.loss + self.alpha_d * disc_loss
            total_loss.backward()

        else:
            disc_logits, disc_loss = self._discriminate(emb, domain_ids)

            # compute feature extractor loss as the KL divergence btw
            # disc output prob distr and uniform distr.
            disc_log_prob = F.log_softmax(disc_logits, dim=1)
            fe_loss = out.loss - self.alpha_d * torch.mean(disc_log_prob)

            # compute grad
            disc_loss.backward(inputs=list(self.disc.parameters()),
                               retain_graph=True)
            fe_loss.backward(inputs=list(self.model.parameters()))

        self.optimizer.step()
        self.opt_disc.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # also return discriminator loss
        return AlgOut(loss={'loss': out.loss, 'disc_loss': disc_loss},
                      logits=out.logits)


class MLDG(ERM):
    def __init__(self, model, device, n_samples_per_domain, alpha_meta, args):
        super().__init__(model, device, args)

        # number of samples per domain in a batch during training
        self.n_samples_per_domain = n_samples_per_domain

        # weight of the meta loss
        self.alpha_meta = alpha_meta

        # optimizer args for inner opt
        self.inner_lr = args.lr2
        self.inner_wd = args.wd2

    def update(self, batch):
        """
        For MLDG, DomainSampler is used in data loader. Therefore, `batch`
        is assumed to contain n_src_dom * n_samples_per_dom samples. And the
        samples are already ordered by domains.
        """
        nd = len(batch[1]) // self.n_samples_per_domain
        loss, meta_loss = 0., 0.
        self.optimizer.zero_grad()

        # initialize grads first
        for p in self.model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # use i-th domain as meta-train
        # use j-th domain as meta-test
        for xi, yi, xj, yj in random_pair_batch(
                batch, self.n_samples_per_domain):

            # clone the original model first
            inner_net = copy.deepcopy(self.model)
            inner_opt = optim.Adam(inner_net.parameters(),
                                   lr=self.inner_lr,
                                   weight_decay=self.inner_wd)
            inner_opt.zero_grad()

            # step using meta-train samples
            inner_obj = F.cross_entropy(inner_net(xi), yi.to(self.device))
            inner_obj.backward()
            inner_opt.step()

            # clone the gradient to original model
            for p_tgt, p_src in zip(self.model.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.add_(p_src.grad / nd)

            # calculate the first-order meta-gradient
            meta_obj = F.cross_entropy(inner_net(xj), yj.to(self.device))
            meta_grad = torch.autograd.grad(meta_obj,
                                            inner_net.parameters(),
                                            allow_unused=True)

            # clone the meta-gradient to original model
            for p_tgt, g in zip(self.model.parameters(), meta_grad):
                if g is not None:
                    p_tgt.grad.add_(self.alpha_meta * g / nd)

            # accumulate loss
            loss += inner_obj.detach()
            meta_loss += meta_obj.detach()

        # after accumulating the grads, perform step
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        with torch.no_grad():
            logits = self.predict(batch[0]).logits

        return AlgOut(loss={'loss': loss / nd, 'meta_loss': meta_loss / nd},
                      logits=logits)

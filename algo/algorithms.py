import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from algo.utils import AlgOut, get_optimizer, get_scheduler, random_pair_batch
from model.models import GRL, MLP
from utils import freeze_network

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
                 args,
                 num_hidden_d=1,
                 hidden_size_d=1024,
                 alpha_d=1.0,
                 n_iter_d=0,  # num of iter to train disc
                 uniform_d=False,  # force uniform output of disc
                 reweight_d=False,  # reweight disc output
                 train_set=None):  # training set

        super().__init__(model, device, args)
        self.n_domains = n_domains
        self.alpha_d = alpha_d
        self.n_iter_d = n_iter_d
        self.num_hidden_d = num_hidden_d
        self.hidden_size_d = hidden_size_d

        # whether to force disc output to be uniform in the outer optimization
        self.uniform_d = uniform_d

        if train_set is not None:
            self.train_loader_disc = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
            )
            self.train_iter_disc = iter(self.train_loader_disc)
        else:
            self.train_loader_disc = None
            self.train_iter_disc = None

        if train_set is not None and reweight_d:
            # get training set domain ids
            domain_ids = train_set.dataset.domain_ids[train_set.indices]

            # get domain sizes and accordingly domain weights
            _, domain_sizes = torch.unique(domain_ids, return_counts=True)
            inv_sizes = 1. / domain_sizes
            self.weight_d = inv_sizes / torch.sum(inv_sizes)
            print('Reweight enabled for discriminator; weight:', self.weight_d)
        else:
            self.weight_d = torch.ones(3, dtype=torch.float32)

        self.weight_d = self.weight_d.to(device)

        # discriminator
        self._init_disc()

        # optimizer of discriminator
        self.opt_disc = optim.Adam(self.disc.parameters(),
                                   lr=args.lr2,
                                   weight_decay=args.wd2)

    def _init_disc(self):
        self.disc = MLP(self.model.out_size, self.n_domains,
                        self.num_hidden_d, self.hidden_size_d)

    def _discriminate(self, in_emb, domain_ids, y=None):
        disc_logits = self.disc(in_emb)
        disc_loss = F.cross_entropy(disc_logits, domain_ids.to(self.device),
                                    weight=self.weight_d)

        return disc_logits, disc_loss

    def update(self, batch):
        # train the discriminator alone first
        for _ in range(self.n_iter_d):
            self.opt_disc.zero_grad()

            # get unlabeled data
            try:
                unl_batch = next(self.train_iter_disc)
            except StopIteration:
                self.train_iter_disc = iter(self.train_loader_disc)
                unl_batch = next(self.train_iter_disc)

            # feed inputs and get embeddings
            with torch.no_grad():
                x, y, domain_ids = unl_batch
                _ = self.predict(x)
                emb = self.model.emb

            # max discriminator acc
            _, disc_loss = self._discriminate(emb, domain_ids, y=y)
            disc_loss.backward()
            self.opt_disc.step()

        # jointly train classifier, featurizer and discriminator
        self.optimizer.zero_grad()
        self.opt_disc.zero_grad()

        x, y, domain_ids = batch
        out = self.predict(x, y=y)
        emb = self.model.emb  # feature embeddings

        if not self.uniform_d:
            # adversarial training
            _, disc_loss = self._discriminate(GRL.apply(emb), domain_ids, y=y)

            # backward
            total_loss = out.loss + self.alpha_d * disc_loss
            total_loss.backward()

        else:
            disc_logits, disc_loss = self._discriminate(emb, domain_ids, y=y)

            # compute feature extractor loss as the KL divergence btw
            # disc output prob distr and uniform distr.
            disc_log_prob = F.log_softmax(disc_logits, dim=1)
            fe_loss = out.loss - self.alpha_d * torch.mean(disc_log_prob)

            """Deprecated because the `inputs` arg in backward is supported
            only after pytorch 1.8.0

            # compute grad
            disc_loss.backward(inputs=list(self.disc.parameters()),
                               retain_graph=True)
            fe_loss.backward(inputs=list(self.model.parameters()))
            """

            # freeze feature extractor and backward on disc
            freeze_network(self.model)
            disc_loss.backward(retain_graph=True)

            # freeze disc and backward on feature extractor
            freeze_network(self.model, unfreeze=True)
            freeze_network(self.disc)
            fe_loss.backward()

            # unfreeze disc
            freeze_network(self.disc, unfreeze=True)

        self.optimizer.step()
        self.opt_disc.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # also return discriminator loss
        return AlgOut(loss={'loss': out.loss, 'disc_loss': disc_loss},
                      logits=out.logits)


class CDAN(DANN):
    """
    Conditional Adversarial Domain Adaptation, by M. Long et al., NIPS 2018.
    """
    def _init_disc(self):
        self.disc = MLP(self.model.out_size * self.model.num_labels,
                        self.n_domains,
                        self.num_hidden_d,
                        self.hidden_size_d)

    def _discriminate(self, in_emb, domain_ids, y=None):
        N, W, D = len(domain_ids), self.model.num_labels, self.model.out_size
        out_prod = torch.zeros(N, W, D, device=self.device)

        # outer product with hard labels = assign by index
        out_prod[range(N), y.to(self.device)] = in_emb
        out_prod = out_prod.flatten(start_dim=1)

        disc_logits = self.disc(out_prod)
        disc_loss = F.cross_entropy(disc_logits, domain_ids.to(self.device),
                                    weight=self.weight_d)

        return disc_logits, disc_loss


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


class IRM(ERM):
    """Invariant Risk Minimization, by Arjovsky et al., arXiv 2019.
    Code from DomainBed.
    """
    def __init__(self,
                 model,
                 device,
                 n_samples_per_domain,
                 args,
                 penalty_weight=100.0,
                 penalty_anneal_iters=500):
        super().__init__(model, device, args)
        self.n_samples_per_domain = n_samples_per_domain
        self.args = args
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.register_buffer('update_count', torch.tensor([0]))

    def _irm_penalty(self, logits, y):
        dummy_w = torch.tensor(1.).to(self.device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * dummy_w, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * dummy_w, y[1::2])
        grad_1 = torch.autograd.grad(loss_1, [dummy_w], create_graph=True)[0]
        grad_2 = torch.autograd.grad(loss_2, [dummy_w], create_graph=True)[0]
        # estimate of the squared norm of gradient
        norm = torch.sum(grad_1 * grad_2)
        return norm

    def update(self, batch):
        """
        For IRM, DomainSampler is used in data loader. Therefore, `batch`
        is assumed to contain n_src_dom * n_samples_per_dom samples. And the
        samples are already ordered by domains.
        """
        x, y, _ = batch
        assert len(x) % self.n_samples_per_domain == 0
        n_src_domains = len(x) // self.n_samples_per_domain
        y = y.to(self.device)

        # set penalty weight to 1.0 during annealing stage
        penalty_weight = 1.0 if self.update_count < self.penalty_anneal_iters \
            else self.penalty_weight

        # reset Adam when annealing ends because there would be a sharp
        # jump in gradient magnitudes
        if self.update_count == self.penalty_anneal_iters \
                and self.args.optimizer.startswith('Adam'):
            self.optimizer = get_optimizer(self.args.optimizer,
                                           self.model,
                                           self.args)
        self.optimizer.zero_grad()

        # record batch loss and prediction
        loss, logits = 0., []

        for i in range(n_src_domains):
            a = i * self.n_samples_per_domain  # start index
            b = a + self.n_samples_per_domain  # end index
            out = self.predict(x[a:b], y=y[a:b])
            penalty = self._irm_penalty(out.logits, y[a:b])

            # accumulate gradients
            (out.loss + penalty_weight * penalty).backward()

            # record batch loss and prediction
            loss += out.loss.item()
            logits.append(out.logits.detach().clone())

        # rescale the gradients by number of source domains
        factor = 1. / n_src_domains
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad *= factor

        self.optimizer.step()
        self.update_count += 1

        loss /= n_src_domains
        logits = torch.cat(logits, dim=0)
        return AlgOut(loss=loss, logits=logits)

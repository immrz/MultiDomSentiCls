from collections import namedtuple
import torch.optim as optim
import numpy as np
from transformers import get_linear_schedule_with_warmup


AlgOut = namedtuple('AlgOut', ['loss', 'logits'])


def get_optimizer(name, model, args):
    if name == 'SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = optim.SGD(params, lr=args.lr, weight_decay=args.wd,
                        **args.optimizer_kwargs)

    elif name == 'AdamW':
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        opt = optim.AdamW(params, lr=args.lr, **args.optimizer_kwargs)

    elif name == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = optim.Adam(params, lr=args.lr, weight_decay=args.wd,
                         **args.optimizer_kwargs)

    else:
        raise NotImplementedError

    return opt


def get_scheduler(name, optimizer, args):
    if name is None or name.lower() == 'none':
        sche = None

    elif name == 'linear_schedule_with_warmup':
        sche = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=args.n_train_steps,
            **args.scheduler_kwargs
        )

    else:
        raise NotImplementedError

    return sche


def random_pair_batch(batch, ns):
    """
    Used for meta-learning. `batch` contains n_src_dom * n_samples_per_dom
    samples.

    Return paired [(xi, yi), (xj, yj)] where i and j indicates meta-train
    and meta-test batches, respectivley.
    """
    x, y, _ = batch
    assert len(y) % ns == 0
    nd = len(y) // ns
    order = np.random.permutation(nd)

    for idx in range(nd):
        i = order[idx]
        j = order[0] if idx == nd - 1 else order[idx + 1]

        i_slc = slice(i * ns, i * ns + ns)
        j_slc = slice(j * ns, j * ns + ns)

        xi, yi = x[i_slc], y[i_slc]
        xj, yj = x[j_slc], y[j_slc]

        yield xi, yi, xj, yj

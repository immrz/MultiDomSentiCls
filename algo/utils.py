from collections import namedtuple
import torch.optim as optim
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


def get_scheduler(name, optimizer, n_train_steps, args):
    if name is None:
        sche = None

    elif name == 'linear_schedule_with_warmup':
        sche = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            **args.scheduler_kwargs
        )

    else:
        raise NotImplementedError

    return sche

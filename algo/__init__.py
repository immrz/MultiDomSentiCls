from algo.algorithms import ERM, DANN, MLDG


def init_algorithm(name, device, model, datasets, args):
    if name == 'ERM':
        algo = ERM(model, device, args)

    elif name == 'DANN':
        algo = DANN(model, device, args.n_domains, args.num_hidden_d,
                    args.hidden_size_d, args.alpha_d, args.n_iter_d,
                    datasets['train']['loader'], args)

    elif name == 'MLDG':
        algo = MLDG(model, device, args.batch_size, args.alpha_meta, args)

    else:
        raise NotImplementedError

    return algo.to(device)

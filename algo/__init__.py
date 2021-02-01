from algo.algorithms import ERM, DANN


def init_algorithm(name, device, model, args):
    if name == 'ERM':
        algo = ERM(model, device, args)

    elif name == 'DANN':
        algo = DANN(model, device, args.n_domains, args.num_hidden_d,
                    args.hidden_size_d, args.alpha_d, args)

    else:
        raise NotImplementedError

    return algo.to(device)

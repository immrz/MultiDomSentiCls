from algo.algorithms import ERM, DANN, CDAN, MLDG, IRM


def init_algorithm(name, device, model, train_set, args):
    if name == 'ERM':
        algo = ERM(model, device, args)

    elif name == 'DANN':
        algo = DANN(model,
                    device,
                    args.n_domains,
                    args,
                    args.num_hidden_d,
                    args.hidden_size_d,
                    args.alpha_d,
                    args.n_iter_d,
                    args.uniform_d,
                    args.reweight_d,
                    train_set)

    elif name == 'CDAN':
        algo = CDAN(model,
                    device,
                    args.n_domains,
                    args,
                    args.num_hidden_d,
                    args.hidden_size_d,
                    args.alpha_d,
                    args.n_iter_d,
                    args.uniform_d,
                    args.reweight_d,
                    train_set)

    elif name == 'MLDG':
        algo = MLDG(model, device, args.batch_size, args.alpha_meta, args)

    elif name == 'IRM':
        algo = IRM(model, device, args.batch_size, args,
                   penalty_weight=args.penalty_weight,
                   penalty_anneal_iters=args.penalty_anneal_iters)

    else:
        raise NotImplementedError

    return algo.to(device)

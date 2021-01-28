from algo.algorithms import ERM


def init_algorithm(name, device, model, args):
    if name == 'ERM':
        algo = ERM(model, device, args)

    else:
        raise NotImplementedError

    return algo.to(device)

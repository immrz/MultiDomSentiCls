from model.models import BertClassifier


def init_model(name, device, num_labels, args):
    if name.startswith('bert'):
        model = BertClassifier(num_labels, device, bert_type=name,
                               max_token_len=args.max_token_len,
                               need_hook=args.algorithm == 'DANN')
    else:
        raise NotImplementedError

    return model

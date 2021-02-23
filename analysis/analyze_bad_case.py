from analysis.helper import get_model_from_root
from data import ATMFDataset
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import json

from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz


@torch.no_grad()
def generate_bad_case(algorithm, target):
    """
    Collect the bad cases of an algorithm w.r.t. some target domain, and save
    them in json format.
    """
    bert = get_model_from_root(f'{algorithm}_tgt_{target}').to('cuda:0')
    bert.eval()

    # get test set
    data_path = '/home/v-runmao/projects/ATMF/data/data_root/preprocessed.csv'
    full_data = ATMFDataset(data_path, target)
    test_set = full_data.get_subset('test')
    loader = DataLoader(test_set, batch_size=128, shuffle=False,
                        pin_memory=True, num_workers=4, drop_last=False)

    # inference
    bad_cases = []
    for x, y, _ in tqdm(loader):
        logits = bert(x)
        prob = torch.nn.functional.softmax(logits, dim=1)
        pred = torch.max(logits, 1)[1].cpu()
        wrong_idx = torch.nonzero(pred != y, as_tuple=True)[0]

        bad_cases.extend([{'text': x[i], 'label': y[i].item(),
                           'confidence': prob[i, y[i]].item()}
                          for i in wrong_idx])

    # output bad cases
    with open(f'bad_cases/{algorithm}_{target}.json', 'w') as fo:
        json.dump(bad_cases, fo, ensure_ascii=False, indent=2)

    acc = 1 - len(bad_cases) / len(test_set)
    print(f'Acc: {acc*100:.2f}%')


def construct_input_ref_pair(tokenizer, text):
    """Constructs text and reference input ids
    """
    input_ids = tokenizer(text, truncation=True, max_length=512)['input_ids']

    assert len(input_ids) >= 2
    ref_ids = [tokenizer.cls_token_id] \
        + [tokenizer.pad_token_id] * (len(input_ids)-2) \
        + [tokenizer.sep_token_id]

    return torch.tensor([input_ids]), torch.tensor([ref_ids])


def construct_aux_param(input_ids):
    """Construct token type ids, attention masks and position ids.
    """
    seq_len = input_ids.shape[1]
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    return token_type_ids, position_ids, attention_mask


def int_to_label(i):
    if type(i) == str:
        return i
    return 'pos' if i == 1 else 'neg'


def interprete_texts(insts, lig, model, tokenizer):
    res = []

    for inst in insts:
        y, text, confidence = inst['label'], inst['text'], inst['confidence']
        model.zero_grad()

        input_ids, ref_ids = construct_input_ref_pair(tokenizer, text)
        tti, pi, am = construct_aux_param(input_ids)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_ids,
            additional_forward_args=(tti, pi, am),
            target=1,
            return_convergence_delta=True,
        )

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        all_tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze().tolist()
        )

        vis = viz.VisualizationDataRecord(attributions,
                                          1 - confidence,
                                          int_to_label(1 - y),
                                          int_to_label(y),
                                          'NA',
                                          attributions.sum(),
                                          all_tokens,
                                          delta)
        res.append(vis)

    viz.visualize_text(res)


def visualize(algorithm, target, n=10, sort=False, max_len=None):
    with open(f'bad_cases/{algorithm}_{target}.json', 'r') as fi:
        bad_cases = json.load(fi)

    bert = get_model_from_root(f'{algorithm}_tgt_{target}').to('cuda:0')
    bert.eval()

    # specify layer to which gradients are computed
    layer = bert.bfsc.bert.embeddings

    # define forward function
    def f(input_ids, token_type_ids, position_ids, attention_mask):
        return bert({'input_ids': input_ids,
                     'token_type_ids': token_type_ids,
                     'position_ids': position_ids,
                     'attention_mask': attention_mask})

    # create lig
    lig = LayerIntegratedGradients(f, layer)

    # sort and slice the bad cases if needed
    if sort:
        bad_cases = sorted(bad_cases, key=lambda x: x['confidence'])
    if max_len:
        bad_cases = [x for x in bad_cases if len(x['text'].split()) <= max_len]
    bad_cases = bad_cases[:n]

    interprete_texts(bad_cases, lig, bert, bert.tknz)

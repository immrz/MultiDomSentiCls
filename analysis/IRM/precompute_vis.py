from analysis.helper import get_model_from_root, get_pred_and_prob
from analysis.analyze_bad_case import construct_aux_param, construct_input_ref_pair
import pandas as pd

import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz

from typing import List, Dict, Tuple
import os
import pickle

import transformers
transformers.logging.set_verbosity_error()


def get_sliced_data(path: str,
                    chunk_id: int,
                    chunk_size: int,
                    has_negation: bool = None,
                    max_len: int = -1,
                    domain: str = None) -> Tuple[pd.DataFrame, Tuple[int, int]]:
    """Get data with specified `domain' from `path'. If `domain' is None, get all domains.
    Sentences whose length is greater than `max_len' are excluded to avoid GPU OutOfMemory.
    After filtering, data indexed from chunk_id*chunk_size to (chunk_id+1)*chunk_size is retrieved.
    """
    data = pd.read_csv(path)

    # whether texts with or without negations
    if has_negation is not None:
        data = data[data['has_negation'] == has_negation]

    # which domain
    if domain is not None:
        data = data[data['domain'] == domain]

    # exclude too long sentences
    if max_len > 0:
        data = data[data['num_token'] <= max_len]

    # determine the start and end position of retrieved data
    N = len(data)
    if chunk_id * chunk_size >= N:
        chunk_id = 0  # display the first chunk
    start = chunk_id * chunk_size
    end = min(N, (chunk_id + 1) * chunk_size)
    data = data.iloc[start:end]

    return data.copy(), (start, end)


def get_visualization(algorithm: str,
                      target: str,
                      data: pd.DataFrame,
                      root: str = '/home/v-runmao/projects/DomShift-ATMF/pt/all_best') -> List[Dict]:

    # get the classifier
    algo_root = os.path.join(root, f'{algorithm}_BEST')
    model = get_model_from_root(algo_root, f'{algorithm}_tgt_{target}').to('cuda:0')
    assert model.device == 'cuda:0', 'Model is not on GPU!'

    # prerequisites
    model.eval()
    tokenizer = model.tknz  # tokenizer
    layer = model.bfsc.bert.embeddings  # the embedding layer to interpret on
    data = data.copy()  # will modify data inplace

    # predict the texts if they are not predicted yet
    key_pred, key_pred_prob = f'{algorithm}_pred', f'{algorithm}_pred_prob'
    if (key_pred not in data.columns) or (key_pred_prob not in data.columns):
        pred, pred_prob = get_pred_and_prob(model, data['text'].tolist())
        data[key_pred] = pred
        data[key_pred_prob] = pred_prob

    # define the forward function
    def f(input_ids, token_type_ids, position_ids, attention_mask):
        return model({'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'position_ids': position_ids,
                      'attention_mask': attention_mask})

    # compute visualization results
    lig = LayerIntegratedGradients(f, layer)
    res = []
    for ix in data.index:
        row = data.loc[ix]
        if type(ix) != int:  # numpy.int64
            ix = ix.item()

        model.zero_grad()

        input_ids, ref_ids = construct_input_ref_pair(tokenizer, row['text'])
        tti, pi, am = construct_aux_param(input_ids)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_ids,
            additional_forward_args=(tti, pi, am),
            target=1,
            return_convergence_delta=True,
        )

        attributions, delta = attributions.detach(), delta.detach().item()

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().numpy()

        all_tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze().tolist()
        )

        vis = viz.VisualizationDataRecord(attributions,
                                          row[key_pred_prob],
                                          row[key_pred],
                                          row['label'],
                                          'NA',
                                          attributions.sum(),
                                          all_tokens,
                                          delta)
        res.append({'vis': vis,
                    'index': ix,
                    'text': row['text'],
                    'label': row['label'],
                    'domain': row['domain'],
                    'pred': row[key_pred],
                    'pred_prob': row[key_pred_prob]})

        torch.cuda.empty_cache()

    return res


def box_info(algorithm, target, domain, max_len, start, end, box_width=90):
    msg = f'|    Model: {algorithm}_tgt_{target}; Domain: {domain}; Max Len: {max_len}; Start Index: {start}; End Index: {end}'
    print('+' + '-' * (box_width-2) + '+')
    print(msg + ' '*(box_width-1-len(msg)) + '|')
    print('+' + '-' * (box_width-2) + '+')


def visualize(algorithm: str,
              target: str,
              domain: str,
              data_path: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM/postprocessed.csv',
              chunk_id: int = 0,
              chunk_size: int = 20,
              max_len: int = -1) -> None:

    data, (start, end) = get_sliced_data(path=data_path,
                                         chunk_id=chunk_id,
                                         chunk_size=chunk_size,
                                         has_negation=True,
                                         max_len=max_len,
                                         domain=domain)
    res = get_visualization(algorithm, target, data)

    # print message and return the data
    box_info(algorithm, target, domain, max_len, start, end)
    viz.visualize_text([r['vis'] for r in res])


def save(algorithm: str,
         target: str,
         max_len: int,
         data_path: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM/postprocessed.csv',
         save_root: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM') -> None:

    fname = f'{algorithm}_tgt_{target}_{max_len}_vis.pkl'
    # setting chunk_id and chunk_size to large numbers simply means using all the data
    data, _ = get_sliced_data(path=data_path,
                              chunk_id=100,
                              chunk_size=1000000,
                              has_negation=True,
                              max_len=max_len)
    print(f'With Max Len {max_len}, the data size is {len(data)}.')
    res = get_visualization(algorithm, target, data)
    with open(os.path.join(save_root, fname), 'wb') as fo:
        pickle.dump(res, fo)


def load_and_visualize(algorithm: str,
                       target: str,
                       max_len: int,
                       domain: str,
                       save_root: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM',
                       label: str = None,
                       t_or_f: str = None,
                       chunk_id: int = 0,
                       chunk_size: int = 20) -> List[Tuple[int, str]]:

    fname = f'{algorithm}_tgt_{target}_{max_len}_vis.pkl'
    with open(os.path.join(save_root, fname), 'rb') as fi:
        res = pickle.load(fi)

    # keep the specified domain
    res = [r for r in res if r['domain'] == domain]

    # whether only samples with particular label
    if label is not None:
        assert label in ['positive', 'negative']
        res = [r for r in res if r['label'] == label]

    # whether only samples that are done wrong or right
    if t_or_f is not None:
        assert t_or_f in ['true', 'false']
        if t_or_f == 'true':
            res = [r for r in res if r['label'] == r['pred']]
        else:
            res = [r for r in res if r['label'] != r['pred']]

    # slice
    N = len(res)
    if chunk_id * chunk_size >= N:
        chunk_id = 0  # display the first chunk
    start = chunk_id * chunk_size
    end = min(N, (chunk_id + 1) * chunk_size)
    res = res[start:end]

    # print message and return the data
    box_info(algorithm, target, domain, max_len, start, end)
    viz.visualize_text([r['vis'] for r in res])

    return [(r['index'], r['text']) for r in res]


def visualize_adversary(algorithm: str,
                        target: str,
                        adversary: List[Tuple[int, str]],
                        data_path: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM/postprocessed.csv',
                        save_root: str = '/home/v-runmao/projects/DomShift-ATMF/analysis/IRM',
                        comparison: bool = True,
                        max_len: int = 100) -> None:
    idx, texts = zip(*adversary)
    data = pd.read_csv(data_path).loc[list(idx)].copy()
    data.drop(['ERM_pred', 'ERM_pred_prob'], axis=1, inplace=True)

    # compute the visualization of adversarial examples
    data['text'] = texts
    adv_res = get_visualization(algorithm, target, data)

    # load the visualization of original texts
    if comparison:
        fname = f'{algorithm}_tgt_{target}_{max_len}_vis.pkl'
        with open(os.path.join(save_root, fname), 'rb') as fi:
            all_res = pickle.load(fi)
        origin_res = [r for r in all_res if r['index'] in idx]
        adv_res = [r for pair in zip(origin_res, adv_res) for r in pair]

    # visualize
    viz.visualize_text([r['vis'] for r in adv_res])


if __name__ == '__main__':
    for algorithm in ['ERM', 'IRM']:
        for target in 'ATMF':
            print(f'{algorithm} | {target}')
            save(algorithm=algorithm, target=target, max_len=100)
            print('finished!')
    print('Thanks for your patience!')

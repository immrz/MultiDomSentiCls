import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from typing import List, Dict, Union


class GRL(torch.autograd.Function):
    """The gradient reversal layer. Same as identical function during forward,
    but reverse gradients during backward.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return -grad


class BertClassifier(nn.Module):
    def __init__(self, num_labels, device, bert_type='bert-base-uncased',
                 max_token_len=512):

        super().__init__()
        self.device = device
        self.max_token_len = max_token_len
        self.emb = None  # used to store BERT feature

        self.tknz = BertTokenizer.from_pretrained(bert_type)
        self.bfsc = BertForSequenceClassification.from_pretrained(
            bert_type, num_labels=num_labels)

        # register the hook which helps store input to classifier layer
        self.bfsc.classifier.register_forward_hook(self.get_input_hook())

    def get_input_hook(self):
        def hook(module, input, output):
            self.emb = input
        return hook

    def forward(self, x: Union[List[str], Dict[str, torch.Tensor]]):
        """
        x is either a list of sentences, or already a dict from strings
        (`input_ids` etc) to pytorch Tensors.
        """
        if isinstance(x, (list, tuple)):
            x = self.tknz(x,
                          padding='max_length',
                          truncation=True,
                          max_length=self.max_token_len,
                          return_tensors='pt')

        # copy to device
        x = {k: v.to(self.device) for k, v in x.items()}

        # forward and return logits
        out = self.bfsc(**x)
        return out.logits

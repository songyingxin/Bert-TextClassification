# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Conv import Conv1d
from Models.Linear import Linear

class BertCNN(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    ```
    """

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        super(BertCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        encoded_layers = self.dropout(encoded_layers)

        encoded_layers = encoded_layers.permute(0, 2, 1)

        conved = self.convs(encoded_layers)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))

        logits = self.classifier(cat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

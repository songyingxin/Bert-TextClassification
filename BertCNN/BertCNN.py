# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Conv import Conv1d
from Models.Linear import Linear

class BertCNN(BertPreTrainedModel):

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        super(BertCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        
        encoded_layers = self.dropout(encoded_layers)

        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim=768, seq_len]

        conved = self.convs(encoded_layers)
        # conved 是一个列表， conved[0]: [batch_size, filter_num, *]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # pooled 是一个列表， pooled[0]: [batch_size, filter_num]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [batch_size, filter_num * len(filter_sizes)]

        logits = self.classifier(cat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

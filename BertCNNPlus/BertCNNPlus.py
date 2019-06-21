# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Conv import Conv1d
from Models.Linear import Linear


class BertCNNPlus(BertPreTrainedModel):

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        super(BertCNNPlus, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters + config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, hidden_state = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        # hidden_state: [batch_size, bert_dim=768]

        encoded_layers = self.dropout(encoded_layers)
        hidden_state = self.dropout(hidden_state)

        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim, seq_len]
        
        conved = self.convs(encoded_layers)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = torch.cat(pooled, dim=1)
        cat = torch.cat([cat, hidden_state], dim=1)

        logits = self.classifier(cat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

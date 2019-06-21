# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BertHAN(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertHAN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        # flat_input_ids: [batch_size * sentence_num, seq_len]

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_token_type_ids: [batch-size * sentence_num, seq_len]

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # flat_attention_mask: [batch_size * sentence_num, seq_len]

        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        # pooled_output: [batch_size * sentence_num, bert_dim]

        pooled_output = self.dropout(pooled_output)

        pooled_output = pooled_output.view(input_ids.size(0), input_ids.size(1), pooled_output.size(-1))
        # pooled_output: [batch_size, sentence_num, bert_dim]
        pooled_output = pooled_output.permute(0, 2, 1)
        # pooled_output: [batch_size, bert_dim, sentence_num]

        pooled_output = F.max_pool1d(pooled_output, pooled_output.size()[2]).squeeze(2)
        # pooled_output: [batch_size, bert_dim]

        logits = self.classifier(pooled_output)
        # logits: [batch_size, num_labels]

        reshape_logits = logits.view(-1, self.num_labels)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Linear import Linear


class BertRCNN(BertPreTrainedModel):

    def __init__(self, config, num_labels, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertRCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,
                           bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.W2 = Linear(config.hidden_size + 2 * rnn_hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        outputs, _= self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]

        x = torch.cat((outputs, encoded_layers), 2)
        # x: [batch_size, seq_len, rnn_hidden_size * 2 + bert_dim]

        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        # y2: [batch_size, rnn_hidden_size * 2, seq_len]

        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)

        logits = self.classifier(y3)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x

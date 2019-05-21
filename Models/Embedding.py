import torch.nn as nn
import torch.nn.functional as F
import torch

from .Highway import Highway

class Embedding(nn.Module):
    """
    word and char embedding

    Input shape: word_emb=(batch_size,sentence_length,emb_size) char_emb=(batch_size,sentence_length,word_length,emb_size)
    Output shape: y= (batch_size,sentence_length,word_emb_size+char_emb_size)
    """

    def __init__(self, highway_layers, word_dim, char_dim):
        super(Embedding, self).__init__()
        self.highway = Highway(highway_layers, word_dim + char_dim)

    def forward(self, word_emb, char_emb):
        char_emb, _ = torch.max(char_emb, 2)

        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)

        return emb

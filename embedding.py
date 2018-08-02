import torch
import torch.nn as nn
from numpy import prod

class WordEmbedding(nn.Module):
    def __init__(self, word_embedding, dropout):
        super(WordEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args, **kwargs):
        words = kwargs['words']
        x = self.word_embedding(words)
        return self.dropout(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class TagEmbedding(nn.Module):
    def __init__(self, word_embedding, dropout):
        super(TagEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args, **kwargs):
        tags = kwargs['tags']
        x = self.word_embedding(tags)
        return self.dropout(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class WordTagEmbedding(nn.Module):
    def __init__(self, word_embedding, tag_embedding, dropout):
        super(WordTagEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.tag_embedding = tag_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args, **kwargs):
        words, tags = kwargs['words'], kwargs['tags']
        words = self.word_embedding(words)
        tags = self.tag_embedding(tags)
        x = torch.cat((words, tags), dim=-1)
        return self.dropout(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)

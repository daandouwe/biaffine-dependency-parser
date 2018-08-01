import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, word_embedding, dropout):
        super(WordEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, **kwargs):
        words = kwargs['words']
        x = self.word_embedding(words)
        return self.dropout(x)

class WordTagEmbedding(nn.Module):
    def __init__(self, word_embedding, tag_embedding, dropout):
        super(WordTagEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.tag_embedding = tag_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, **kwargs):
        words, tags = kwargs['words'], kwargs['tags']
        words = self.word_embedding(words)
        tags = self.tag_embedding(tags)
        x = torch.cat((words, tags), dim=-1)
        return self.dropout(x)

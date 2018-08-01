import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, wrap

class MLP(nn.Module):
    """Module for an MLP with dropout."""
    def __init__(self, input_size, layer_size, depth, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(input_size, layer_size))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            input_size = layer_size

    def forward(self, x):
        return self.layers(x)


class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_uniform(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)

    # TODO: add collumns of ones to Rh and Rd for biases.



class RecurrentCharEmbedding(nn.Module):
    """Simple RNN based encoder for character-level word embeddings.

    Based on: https://github.com/bastings/parser/blob/extended_parser/parser/nn.py"""
    def __init__(self, nchars, output_size, padding_idx, hidden_size=None, emb_dim=None, dropout=0.33, bi=True):
        super(RecurrentCharEmbedding, self).__init__()
        # Default values for encoder.
        emb_dim = output_size if emb_dim is None else emb_dim
        hidden_size = output_size if hidden_size is None else hidden_size

        self.embedding = nn.Embedding(nchars, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, dropout=dropout, bidirectional=bi)

        rnn_dim = hidden_size * 2 if bi else hidden_size
        self.linear = nn.Linear(rnn_dim, output_size, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        cuda = torch.cuda.is_available()

        batch_size, sent_len, word_len = x.shape
        x = x.view(-1, word_len)

        # Sort x by word length.
        lengths = (x != PAD_INDEX).long().sum(-1)
        sorted_lengths, sort_idx = lengths.sort(0, descending=True)
        sort_idx = sort_idx.cuda() if cuda else sort_idx
        x = x[sort_idx]

        # Remove the rows (i.e. words) from x that consist entirely of PAD_INDEX.
        non_padding_idx = (sorted_lengths != 0).long().sum().data[0]
        num_all_pad = x.size(0) - non_padding_idx
        x = x[:non_padding_idx]
        sorted_lengths = sorted_lengths[:non_padding_idx]

        # Embed chars and pack for rnn input.
        x = self.embedding(x)
        x = self.dropout(x)
        sorted_lengths = [i for i in sorted_lengths.data]
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)

        # RNN computation.
        out, _ = self.rnn(x)

        # Unpack and keep only final embedding.
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        sorted_lengths = wrap(sorted_lengths) - 1
        sorted_lengths = sorted_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.size(-1))
        sorted_lengths = sorted_lengths.cuda() if cuda else sorted_lengths
        out = torch.gather(out, 1, sorted_lengths).squeeze(1)

        # Project rnn output states to proper embedding dimension.
        out = self.relu(out)
        out = self.linear(out)

        # Put back zero vectors for the pad words that we removed.
        pad_embeddings = Variable(torch.zeros(num_all_pad, out.size(-1)))
        pad_embeddings = pad_embeddings.cuda() if cuda else pad_embeddings
        out = torch.cat([out, pad_embeddings])

        # Put everything back into the original order.
        pairs = list(zip(sort_idx.data, range(sort_idx.size(0))))
        undo_sort_idx = [pair[1] for pair in sorted(pairs, key=lambda t: t[0])]
        undo_sort_idx = wrap(undo_sort_idx)
        undo_sort_idx = undo_sort_idx.cuda() if cuda else undo_sort_idx
        out = out[undo_sort_idx]
        out = out.view(batch_size, sent_len, out.size(-1))

        return out

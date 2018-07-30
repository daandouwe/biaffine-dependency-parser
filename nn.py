import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, wrap

class RecurrentCharEmbedding(nn.Module):
    """Simple RNN based encoder for character-level word embeddings.

    Based on: https://github.com/bastings/parser/blob/extended_parser/parser/nn.py"""
    def __init__(self, nchars, emb_dim, hidden_size, output_dim, padding_idx, dropout=0.33, bi=True):
        super(RecurrentCharEmbedding, self).__init__()

        self.embedding = nn.Embedding(nchars, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, dropout=dropout, bidirectional=bi)

        rnn_dim = hidden_size * 2 if bi else hidden_size
        self.linear = nn.Linear(rnn_dim, output_dim, bias=False)

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

        # Remove all rows from x that are all all PAD_INDEX.
        non_padding_idx = (sorted_lengths != 0).long().sum().data[0]
        num_all_pad = x.size(0) - non_padding_idx
        x = x[:non_padding_idx]
        sorted_lengths = sorted_lengths[:non_padding_idx]

        # Embed chars and pack for rnn input
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
        out = out[undo_sort_idx]
        out = out.view(batch_size, sent_len, out.size(-1))

        return out

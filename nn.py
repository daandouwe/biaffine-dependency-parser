import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy import prod

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

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


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

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


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
        # Reshape so that the characters are the only sequences.
        x = x.view(-1, word_len) # (batch_size * sent_len, word_len)

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
        if num_all_pad > 0:
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

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class ResidualConnection(nn.Module):
    """A residual connection with dropout."""
    def __init__(self, layer, dropout):
        super(ResidualConnection, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.layer(x))

class SimpleConvolutionalCharEmbedding(nn.Module):
    def __init__(self, nchars, output_size, padding_idx, num_conv=3, kernel_size=3, emb_dim=None,
                hidden_size=None, activation='ReLU', dropout=0.):
        super(SimpleConvolutionalCharEmbedding, self).__init__()
        emb_dim = output_size if emb_dim is None else emb_dim
        hidden_size = output_size if hidden_size is None else hidden_size
        # Make sure kernel_size is odd.
        assert kernel_size / 2 > kernel_size // 2
        # Padding to keep shape constant
        padding = kernel_size // 2
        act_fn = getattr(nn, activation)
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(nchars, emb_dim, padding_idx=padding_idx)
        input_size = emb_dim
        layers = nn.Sequential()
        for i in range(num_conv):
            conv = nn.Conv1d(input_size, output_size, kernel_size, padding=padding)
            layers.add_module('res_{}'.format(i),
                    ResidualConnection(conv, dropout))
            layers.add_module('{}_{}'.format(activation, i),
                    act_fn())
            input_size = output_size
        self.layers = layers

    def forward(self, x):
        """Expect input of shape (batch, seq, emb)."""
        batch_size, sent_len, word_len = x.shape
        x = x.view(-1, word_len) # (batch * sent, word)
        mask = (x != self.padding_idx).float()
        x = self.embedding(x)   # (batch * sent, word, emb)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = mask * x
        x = x.transpose(1, 2)   # (batch * sent, emb, word)
        x = self.layers(x)      # (batch * sent, emb, word)
        x = x.mean(-1)          # (batch * sent, emb)
        x = x.contiguous().view(batch_size, sent_len, x.size(-1)) # (batch, sent, emb)
        return x

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class HighwayNetwork(nn.Module):
    def __init__(self, input_size, activation='ReLU'):
        super(HighwayNetwork, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, 1)
        self.act_fn = getattr(nn, activation)()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.sigmoid(self.gate(x))
        out = self.act_fn(self.linear(x))
        return t * out + (1 - t) * x


class ConvolutionalCharEmbedding(nn.Module):
    """Convolutional character embedding following https://arxiv.org/pdf/1508.06615.pdf."""
    def __init__(self, nchars, padding_idx, emb_dim=15, filter_factor=25, activation='Tanh', dropout=0.):
        super(ConvolutionalCharEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(nchars, emb_dim, padding_idx=padding_idx)

        filter_size = lambda kernel_size: filter_factor * kernel_size
        self.output_size = sum(map(filter_size, range(1, 7)))
        self.conv1 = nn.Conv1d(emb_dim, filter_size(1), kernel_size=1)
        self.conv2 = nn.Conv1d(emb_dim, filter_size(2), kernel_size=2)
        self.conv3 = nn.Conv1d(emb_dim, filter_size(3), kernel_size=3)
        self.conv4 = nn.Conv1d(emb_dim, filter_size(4), kernel_size=4)
        self.conv5 = nn.Conv1d(emb_dim, filter_size(5), kernel_size=5)
        self.conv6 = nn.Conv1d(emb_dim, filter_size(6), kernel_size=6)

        self.act_fn = getattr(nn, activation)()

        self.pool = nn.AdaptiveMaxPool1d(1) # Max pooling over time.

        self.highway = HighwayNetwork(self.output_size)

    def forward(self, x):
        """Expect input of shape (batch, sent_len, word_len)."""
        # Preprocessing of character batch.
        batch_size, sent_len, word_len = x.shape
        x = x.view(-1, word_len) # (batch * sent, word)
        mask = (x != self.padding_idx).float()
        x = self.embedding(x)   # (batch * sent, word, emb)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = mask * x
        x = x.transpose(1, 2)   # (batch * sent, emb, word)

        # Ready for input
        f1 = self.pool(self.act_fn(self.conv1(x))).squeeze(-1)
        f2 = self.pool(self.act_fn(self.conv2(x))).squeeze(-1)
        f3 = self.pool(self.act_fn(self.conv3(x))).squeeze(-1)
        f4 = self.pool(self.act_fn(self.conv4(x))).squeeze(-1)
        f5 = self.pool(self.act_fn(self.conv5(x))).squeeze(-1)
        f6 = self.pool(self.act_fn(self.conv6(x))).squeeze(-1)

        f = torch.cat([f1, f2, f3, f4, f5, f6], dim=-1)

        f = self.highway(f)

        return f.contiguous().view(batch_size, sent_len, f.size(-1)) # (batch, sent, emb)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    import argparse
    from data import Corpus

    parser = argparse.ArgumentParser(description='Biaffine graph-based dependency parser')

    parser.add_argument('--data', type=str, default='~/data/stanford-ptb/',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default='vocab/train',
                        help='location of the preprocessed vocabulary')
    parser.add_argument('--char', action='store_true',
                        help='character embeddings')
    parser.add_argument('--disable_length_ordered', action='store_false',
                        help='do not order sentences by length so batches have more padding')
    args = parser.parse_args()

    corpus = Corpus(data_path=args.data, vocab_path=args.vocab, char=args.char)
    batches = corpus.train.batches(4, length_ordered=args.disable_length_ordered)

    words, tags, heads, labels = next(batches)

    emb_dim = 100
    embedding = ConvolutionalCharEmbedding(len(corpus.dictionary.w2i), emb_dim)

    x = embedding(words)
    print(x)

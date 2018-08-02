import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import prod

from nn import ResidualConnection

class RecurrentEncoder(nn.Module):
    """A simple RNN based sentence encoder."""
    def __init__(self, rnn_type, input_size, hidden_size, num_layers,
                 batch_first, dropout, bidirectional,
                 use_cuda=False, hidden_init='zeros', train_hidden_init=False):
        super(RecurrentEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first

        assert rnn_type in ('LSTM', 'GRU', 'RNN')
        self.rnn_type = rnn_type
        args = input_size, hidden_size, num_layers, batch_first, dropout, bidirectional
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers,
                                            batch_first=batch_first,
                                            dropout=dropout,
                                            bidirectional=bidirectional)

        assert hidden_init in ('zeros', 'randn') # availlable initialization methods.
        self.hidden_init = getattr(torch, hidden_init)
        self.train_hidden_init = train_hidden_init # TODO: make initial hidden trainable.

        self.cuda = use_cuda

    def get_hidden(self, batch):
        args = self.num_layers * self.num_directions, batch, self.hidden_size
        use_cuda = torch.cuda.is_available()
        if self.rnn_type == 'LSTM':
            h0 = Variable(self.hidden_init(*args)) # (num_layers * rections, batch, hidden_size)
            c0 = Variable(self.hidden_init(*args)) # (num_layers * rections, batch, hidden_size)
            if use_cuda:
                h0, c0 = h0.cuda(), c0.cuda()
            return h0, c0
        else:
            h0 = Variable(self.hidden_init(*args)) # (num_layers * rections, batch, hidden_size)
            if use_cuda:
                h0 = h0.cuda()
            return h0

    def forward(self, x, lengths):
        batch = x.size(0) if self.batch_first else x.size(1)
        # RNN computation.

        h0 = self.get_hidden(batch)
        out, _ = self.rnn(x, h0)
        # batch = x.size(0) if self.batch_first else x.size(1)
        #
        # # Sort x by word length.
        # sorted_lengths, sort_idx = lengths.sort(0, descending=True)
        # sort_idx = sort_idx.cuda() if self.cuda else sort_idx
        # print(sort_idx.shape)
        # sort_idx = sort_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(-1))
        # print(sort_idx.shape)
        # quit()
        # x = x[sort_idx]
        #
        # # Embed chars and pack for rnn input.
        # sorted_lengths_list = list(sorted_lengths.data)
        # x = pack_padded_sequence(x, sorted_lengths_list, batch_first=self.batch_first)
        #
        # # RNN computation.
        # h0 = self.get_hidden(batch)
        # out, _ = self.rnn(x, h0)
        #
        # # Unpack and keep only final embedding.
        # out, _ = pad_packed_sequence(out, batch_first=self.batch_first)
        # print(out.shape)
        # sorted_lengths = sorted_lengths - 1
        # sorted_lengths = sorted_lengths.unsqueeze(-1).unsqueeze(-1).repeat(out.size(0), 1, out.size(-1))
        # sorted_lengths = sorted_lengths.cuda() if self.cuda else sorted_lengths
        # out = torch.gather(out, 1, sorted_lengths).squeeze(1)
        #
        # # Put everything back into the original order.
        # pairs = list(zip(sort_idx.data, range(sort_idx.size(0))))
        # undo_sort_idx = [pair[1] for pair in sorted(pairs, key=lambda t: t[0])]
        # undo_sort_idx = Variable(torch.LongTensor(undo_sort_idx))
        # out = out[undo_sort_idx]
        return out

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)

class SimpleConvolutionalEncoder(nn.Module):
    def __init__(self, input_size, num_conv, kernel_size, activation='Tanh', dropout=0.):
        super(SimpleConvolutionalEncoder, self).__init__()
        # Make sure kernel_size is odd.
        assert kernel_size / 2 > kernel_size // 2
        # Padding to keep shape constant
        padding = kernel_size // 2
        act_fn = getattr(nn, activation)
        layers = nn.Sequential()
        for i in range(num_conv):
            conv = nn.Conv1d(input_size, input_size, kernel_size, padding=padding)
            layers.add_module('res_{}'.format(i),
                    ResidualConnection(conv, dropout))
            layers.add_module('{}_{}'.format(activation, i),
                    act_fn())
        self.layers = layers

    def forward(self, x, mask):
        """Expect input of shape (batch, seq, emb)."""
        # x = mask * x
        x = x.transpose(1, 2) # (batch, emb, seq)
        x = self.layers(x)
        return x.transpose(1, 2) # (batch, seq, emb)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvolutionalEncoder, self).__init__()
        pass

    def forward(self, x, mask):
        """Expect input of shape (batch, seq, emb)."""
        pass

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)

class NoEncoder(nn.Module):
    """This encoder does nothing."""
    def __init__(self):
        super(NoEncoder, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def num_parameters(self):
        return 0

if __name__ == '__main__':
    m = ConvolutionalEncoder(10, 8, num_conv=3)
    x = Variable(torch.randn(5, 6, 10)) # (batch, seq, emb)
    print(m(x))

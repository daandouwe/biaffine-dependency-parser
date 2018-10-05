import copy

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

        assert hidden_init in ('zeros', 'randn')  # availlable initialization methods.
        self.hidden_init = getattr(torch, hidden_init)
        self.train_hidden_init = train_hidden_init  # TODO: make initial hidden trainable.
        self.cuda = use_cuda

    def get_hidden(self, batch):
        args = self.num_layers * self.num_directions, batch, self.hidden_size
        use_cuda = torch.cuda.is_available()
        if self.rnn_type == 'LSTM':
            h0 = Variable(self.hidden_init(*args))  # (num_layers * directions, batch, hidden_size)
            c0 = Variable(self.hidden_init(*args))  # (num_layers * directions, batch, hidden_size)
            if use_cuda:
                h0, c0 = h0.cuda(), c0.cuda()
            return h0, c0
        else:
            h0 = Variable(self.hidden_init(*args))  # (num_layers * directions, batch, hidden_size)
            if use_cuda:
                h0 = h0.cuda()
            return h0

    def forward(self, x, lengths):
        batch = x.size(0) if self.batch_first else x.size(1)
        # RNN computation.
        h0 = self.get_hidden(batch)
        out, _ = self.rnn(x, h0)
        return out

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class ConvolutionalEncoder(nn.Module):
    """Stacked convolutions with residual connection.

    Similar to the architectures used in https://arxiv.org/pdf/1705.03122.pdf and
    https://arxiv.org/pdf/1611.02344.pdf.
    """
    def __init__(self, input_size, num_conv, kernel_size, activation='Tanh', dropout=0.):
        super(ConvolutionalEncoder, self).__init__()
        assert kernel_size % 2 == 1, 'only odd kernel sizes supported'
        padding = kernel_size // 2 # Padding to keep size constant
        act_fn = getattr(nn, activation)
        layers = nn.Sequential()
        c = copy.deepcopy
        conv = nn.Conv1d(input_size, input_size, kernel_size, padding=padding)
        for i in range(num_conv):
            layers.add_module('res_{}'.format(i),
                    ResidualConnection(c(conv), dropout))
            layers.add_module('{}_{}'.format(activation, i),
                    act_fn())
        self.layers = layers

    def forward(self, x, mask):
        """Expect input of shape (batch, seq, emb)."""
        # x = mask * x
        x = x.transpose(1, 2)  # (batch, emb, seq)
        x = self.layers(x)
        return x.transpose(1, 2)  # (batch, seq, emb)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class NoEncoder(nn.Module):
    """This encoder does nothing."""
    def __init__(self, *args, **kwargs):
        super(NoEncoder, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def num_parameters(self):
        return 0


if __name__ == '__main__':
    m = ConvolutionalEncoder(10, 8, num_conv=3)
    x = Variable(torch.randn(5, 6, 10))  # (batch, seq, emb)
    print(m(x))

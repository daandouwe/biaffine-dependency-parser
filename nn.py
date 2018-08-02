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


class ResidualConnection(nn.Module):
    """A residual connection with dropout."""
    def __init__(self, layer, dropout):
        super(ResidualConnection, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.layer(x))


class HighwayNetwork(nn.Module):
    """A highway network used in the character convolution word embeddings."""
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

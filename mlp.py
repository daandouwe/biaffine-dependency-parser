"""
Source: https://github.com/EelcovdW/NLP1-project/blob/master/src/modules.py
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Module for an MLP with dropout.
    """

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

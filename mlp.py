import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple multilayer perceptron.
    """
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(num_hidden):
            self.layers.add_module('fc_{}'.format(i), nn.Linear(input_dim, hidden_dim))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i), act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i), nn.Dropout(dropout))
            input_dim = hidden_dim

        self.layers.add_module('fc_{}'.format(i+1), nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.layers(x)

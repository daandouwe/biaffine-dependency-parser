import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RecurrentEncoder(nn.Module):
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

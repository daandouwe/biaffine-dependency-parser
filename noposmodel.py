import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mlp import MLP
from biaffine import BiAffine
from data import PAD_INDEX

class BiAffineParser(nn.Module):
    """
    Biaffine parser that does not require POS-tags as input during evaluation.
    Simple idea:
        * POS tags are passed during training, but not inputted to the RNN-encoder.
        * Instead, from the output embeddings of the RNN-encoder, an MLP predicts
          a POS-tag for the word. The loss on this objective is incorporated into
          the total loss.
        * Hopefully this works...
    """
    def __init__(self, word_vocab_size, word_emb_dim,
                 tag_vocab_size, tag_emb_dim, emb_dropout,
                 lstm_hidden, lstm_num_layers, lstm_dropout,
                 mlp_arc_hidden, mlp_lab_hidden, mlp_dropout,
                 num_labels, criterion=nn.CrossEntropyLoss()):
        super(BiAffineParser, self).__init__()

        # Embeddings
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # LSTM
        lstm_input = word_emb_dim
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden,
                            num_layers=lstm_num_layers, batch_first=True,
                            dropout=lstm_dropout, bidirectional=True)

        mlp_input = 2*lstm_hidden
        # POS MLP
        self.pos_mlp = MLP(mlp_input, tag_vocab_size, 2, 'ReLU', mlp_dropout)

        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)
        self.lab_mlp_d = MLP(mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = BiAffine(mlp_lab_hidden, num_labels)

        # Loss criterion
        self.criterion = criterion
        self.pos_criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    def forward(self, words, tags):
        """
        Compute the score matrices for the arcs and labels.
        """
        x = self.word_embedding(words)
        x = self.emb_dropout(x)

        h, _ = self.lstm(x)

        # Use lstm ouput to predict POS-tags
        pos = self.pos_mlp(h)
        pos = pos.view(pos.size(0)*pos.size(1), pos.size(2))
        self.pos_loss = self.pos_criterion(pos, tags.view(-1))
        self.pos_acc = self.pos_accuracy(pos, tags.view(-1))

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)
        lab_h = self.lab_mlp_h(h)
        lab_d = self.lab_mlp_d(h)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)
        return S_arc, S_lab

    def arc_loss(self, S_arc, heads):
        """
        Compute the loss for the arc predictions.
        """
        S_arc = S_arc.transpose(-1, -2)                     # [batch, sent_len, sent_len]
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1)) # [batch*sent_len, sent_len]
        heads = heads.view(-1)                              # [batch*sent_len]
        return self.criterion(S_arc, heads)

    def lab_loss(self, S_lab, heads, labels):
        """
        Compute the loss for the label predictions on the gold arcs (heads).
        """
        heads = heads.unsqueeze(1).unsqueeze(2)             # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)     # [batch, n_labels, 1, sent_len]
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)    # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)                     # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1)) # [batch*sent_len, n_labels]
        labels = labels.view(-1)                            # [batch*sent_len]
        return self.criterion(S_lab, labels)

    def pos_accuracy(self, pred, gold):
        _, pred = pred.max(dim=-1)
        mask = (gold != PAD_INDEX).float()
        acc = mask * (pred == gold).float()
        return acc.mean()

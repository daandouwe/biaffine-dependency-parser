import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import PAD_INDEX
from nn import MLP, BiAffine, RecurrentCharEmbedding
from encoder import RecurrentEncoder
from transformer import make_model

class BiAffineParser(nn.Module):

    def __init__(self, word_vocab_size, word_emb_dim,
                 tag_vocab_size, tag_emb_dim, emb_dropout,
                 rnn_type, rnn_hidden, rnn_num_layers, rnn_dropout,
                 mlp_arc_hidden, mlp_lab_hidden, mlp_dropout,
                 num_labels, criterion, batch_first=True,
                 char=False):
        super(BiAffineParser, self).__init__()
        # Embeddings
        if char:
            self.word_embedding = RecurrentCharEmbedding(word_vocab_size, word_emb_dim, word_emb_dim, word_emb_dim, padding_idx=PAD_INDEX)
        else:
            self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=PAD_INDEX)
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim, padding_idx=PAD_INDEX)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # Encoder
        encoder_input = word_emb_dim + tag_emb_dim
        # RNN
        self.rnn = False
        if self.rnn:
            self.encoder = RecurrentEncoder(rnn_type, encoder_input, rnn_hidden, rnn_num_layers,
                                            batch_first, rnn_dropout, bidirectional=True)
            encoder_dim = 2 * rnn_hidden
        # Tranformer
        self.transformer = True
        if self.transformer:
            self.encoder = make_transformer(d_model=encoder_input, dropout=0.1)
            encoder_dim = encoder_input

        # Arc MLPs
        mlp_input = encoder_dim
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

    def forward(self, words, tags):
        """Compute the score matrices for the arcs and labels."""
        mask = (words != PAD_INDEX).unsqueeze(-2)
        lengths = (words != PAD_INDEX).long().sum(-1)
        aux = mask if self.transformer else lengths

        words = self.word_embedding(words)
        tags = self.tag_embedding(tags)
        x = torch.cat((words, tags), dim=-1)
        x = self.emb_dropout(x)

        h = self.encoder(x, aux)

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)
        lab_h = self.lab_mlp_h(h)
        lab_d = self.lab_mlp_d(h)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)
        return S_arc, S_lab

    def arc_loss(self, S_arc, heads):
        """Compute the loss for the arc predictions."""
        S_arc = S_arc.transpose(-1, -2)                     # [batch, sent_len, sent_len]
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1)) # [batch*sent_len, sent_len]
        heads = heads.view(-1)                              # [batch*sent_len]
        return self.criterion(S_arc, heads)

    def lab_loss(self, S_lab, heads, labels):
        """Compute the loss for the label predictions on the gold arcs (heads)."""
        heads = heads.unsqueeze(1).unsqueeze(2)             # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)     # [batch, n_labels, 1, sent_len]
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)    # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)                     # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1)) # [batch*sent_len, n_labels]
        labels = labels.view(-1)                            # [batch*sent_len]
        return self.criterion(S_lab, labels)

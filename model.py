import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import prod

from data import PAD_INDEX
from embeddings import WordEmbedding, WordTagEmbedding
from nn import MLP, BiAffine, RecurrentCharEmbedding
from encoder import RecurrentEncoder
from transformer import TransformerEncoder

class BiAffineParser(nn.Module):

    def __init__(self, embedding, encoder, encoder_type,
                 mlp_input, mlp_arc_hidden,
                 mlp_lab_hidden, mlp_dropout,
                 num_labels, criterion):
        super(BiAffineParser, self).__init__()

        self.embedding = embedding
        self.encoder = encoder

        self.encoder_type = encoder_type

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
        self.criterion = criterion()

    def forward(self, **kwargs):
        """Compute the score matrices for the arcs and labels."""

        x = self.embedding(**kwargs)

        words = kwargs['words']
        if self.encoder_type == 'transformer':
            aux = (words != PAD_INDEX).unsqueeze(-2) # mask
        if self.encoder_type == 'rnn':
            aux = (words != PAD_INDEX).long().sum(-1) # sentence_lenghts

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

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


def make_model(args, word_vocab_size, tag_vocab_size, num_labels):
    # Embeddings
    # Character embeddins
    if args.use_char:
        if args.char_encoder == 'rnn':
            word_embedding = RecurrentCharEmbedding(word_vocab_size, args.word_emb_dim, padding_idx=PAD_INDEX)
        elif args.char_encoder == 'cnn':
            raise NotImplementedError('CNN character econder not yet implemented.')
        elif args.char_encoder == 'transformer':
            raise NotImplementedError('Transformer character econder not yet implemented.')
    # Word embeddings.
    else:
        word_embedding = nn.Embedding(word_vocab_size, args.word_emb_dim, padding_idx=PAD_INDEX)
        if args.use_glove:
            raise NotImplementedError('GloVe embeddings not yet implemented.')
    # Optional tag embedding
    if args.disable_tags:
        embedding = WordEmbedding(word_embedding, args.emb_dropout)
        encoder_input = args.word_emb_dim
    else:
        tag_embedding = nn.Embedding(tag_vocab_size, args.tag_emb_dim, padding_idx=PAD_INDEX)
        embedding = WordTagEmbedding(word_embedding, tag_embedding, args.emb_dropout)
        encoder_input = args.word_emb_dim + args.tag_emb_dim

    # Encoder
    if args.encoder == 'rnn':
        encoder = RecurrentEncoder(args.rnn_type, encoder_input, args.rnn_hidden, args.rnn_num_layers,
                                   args.batch_first, args.rnn_dropout, bidirectional=True)
        encoder_dim = 2 * args.rnn_hidden
    elif args.encoder == 'cnn':
        raise NotImplementedError('CNN econder not yet implemented.')
    elif args.encoder == 'transformer':
        encoder = TransformerEncoder(encoder_input, args.N, args.d_model, args.d_ff, args.h, dropout=args.transformer_dropout)
        encoder_dim = args.d_model

    # Initialize the model.
    model = BiAffineParser(
                embedding, encoder, args.encoder,
                encoder_dim, args.mlp_arc_hidden,
                args.mlp_lab_hidden, args.mlp_dropout,
                num_labels, nn.CrossEntropyLoss
            )

    # Initialize parameters with Glorot.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

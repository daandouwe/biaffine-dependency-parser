#!/usr/bin/env python
import argparse

from train import train

def main():

    parser = argparse.ArgumentParser(description='Biaffine graph-based dependency parser')

    parser.add_argument('mode', type=str, choices=['train', 'predict'])
    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--data', type=str, default='../../stanford-ptb',
                        help='location of the data corpus')
    data.add_argument('--vocab', type=str, default='vocab/train',
                        help='location of the preprocessed vocabulary')
    data.add_argument('--disable_length_ordered', action='store_false',
                        help='do not order sentences by length so batches have more padding')
    # Embedding
    embed = parser.add_argument_group('Embedding options')
    embed.add_argument('--use_glove', action='store_true',
                        help='use pretrained glove embeddings')
    embed.add_argument('--use_char', action='store_true',
                        help='use character level word embeddings')
    embed.add_argument('--char_encoder', type=str, choices=['rnn', 'cnn', 'transformer'],
                        default='rnn', help='type of character encoder used for word embeddings')
    embed.add_argument('--disable_tags', action='store_false',
                        help='do not use tags as additional input')
    embed.add_argument('--word_emb_dim', type=int, default=300,
                        help='size of word embeddings')
    embed.add_argument('--tag_emb_dim', type=int, default=50,
                        help='size of tag embeddings')
    embed.add_argument('--emb_dropout', type=float, default=0.3,
                        help='dropout used on embeddings')
    # Encoder
    encode = parser.add_argument_group('Encoder options')
    encode.add_argument('--encoder', type=str, choices=['rnn', 'cnn', 'transformer'],
                        default='rnn', help='type of sentence encoder used')
    # RNN encoder arguments
    encode.add_argument('--rnn_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM',
                        help='number of hidden units in RNN')
    encode.add_argument('--rnn_hidden', type=int, default=400,
                        help='number of hidden units in RNN')
    encode.add_argument('--rnn_num_layers', type=int, default=3,
                        help='number of layers')
    encode.add_argument('--batch_first', type=bool, default=True,
                        help='number of layers')
    encode.add_argument('--rnn_dropout', type=float, default=0.3,
                        help='dropout used in rnn')
    # Transformer encoder arguments
    encode.add_argument('--N', type=int, default=6,
                        help='')
    encode.add_argument('--d_model', type=int, default=512,
                        help='')
    encode.add_argument('--d_ff', type=int, default=2048,
                        help='')
    encode.add_argument('--h', type=int, default=8,
                        help='')
    encode.add_argument('--transformer_dropout', type=float, default=0.1,
                        help='dropout used in transformer')
    # Biaffine transformations
    biaff = parser.add_argument_group('Biaffine classifier arguments')
    biaff.add_argument('--mlp_arc_hidden', type=int, default=500,
                        help='number of hidden units in arc MLP')
    biaff.add_argument('--mlp_lab_hidden', type=int, default=100,
                        help='number of hidden units in label MLP')
    biaff.add_argument('--mlp_dropout', type=float, default=0.3,
                        help='dropout used in mlps')
    # Training.
    training = parser.add_argument_group('Training arguments')
    training.add_argument('--multi_gpu', action='store_true',
                        help='enable training on multiple GPUs')
    training.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate')
    training.add_argument('--epochs', type=int, default=10,
                        help='number of epochs of training')
    training.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    training.add_argument('--seed', type=int, default=42,
                        help='random seed')
    training.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    training.add_argument('--print_every', type=int, default=100,
                        help='report interval')
    training.add_argument('--plot_every', type=int, default=100,
                        help='plot interval')
    training.add_argument('--save', type=str,  default='models/model.pt',
                        help='path to save the final model')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)

if __name__ == '__main__':
    main()

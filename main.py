#!/usr/bin/env python
import argparse

from train import train


def main():

    parser = argparse.ArgumentParser(description='Biaffine graph-based dependency parser')

    parser.add_argument('mode', type=str, choices=['train', 'predict'])

    # Data arguments
    data = parser.add_argument_group('Data')
    data.add_argument('--data', type=str, default='~/data/ptb-stanford/',
                        help='location of the data corpus')
    data.add_argument('--vocab', type=str, default='vocab/train',
                        help='location of the preprocessed vocabulary')
    data.add_argument('--disable-length-ordered', action='store_false',
                        help='do not order sentences by length so batches have more padding')

    # Embedding arguments
    embed = parser.add_argument_group('Embedding options')
    embed.add_argument('--use-glove', action='store_true',
                        help='use pretrained glove embeddings')
    embed.add_argument('--use-chars', action='store_true',
                        help='use character level word embeddings')
    embed.add_argument('--char-encoder', type=str, choices=['rnn', 'cnn', 'transformer'],
                        default='cnn', help='type of character encoder used for word embeddings')
    embed.add_argument('--filter-factor', type=int, default=25,
                        help='controls output size of cnn character embedding')
    embed.add_argument('--disable-words', action='store_false',
                        help='do not use words as input')
    embed.add_argument('--disable-tags', action='store_false',
                        help='do not use tags as input')
    embed.add_argument('--word-emb-dim', type=int, default=300,
                        help='size of word embeddings')
    embed.add_argument('--tag-emb-dim', type=int, default=50,
                        help='size of tag embeddings')
    embed.add_argument('--emb-dropout', type=float, default=0.3,
                        help='dropout used on embeddings')

    # Encoder arguments
    encode = parser.add_argument_group('Encoder options')
    encode.add_argument('--encoder', type=str, choices=['rnn', 'cnn', 'transformer', 'none'],
                        default='rnn', help='type of sentence encoder used')

    # RNN encoder arguments
    rnn = parser.add_argument_group('RNN options')
    rnn.add_argument('--rnn-type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM',
                        help='type of rnn')
    rnn.add_argument('--rnn-hidden', type=int, default=400,
                        help='number of hidden units in rnn')
    rnn.add_argument('--rnn-num-layers', type=int, default=3,
                        help='number of layers')
    rnn.add_argument('--batch-first', type=bool, default=True,
                        help='number of layers')
    rnn.add_argument('--rnn-dropout', type=float, default=0.3,
                        help='dropout used in rnn')

    # CNN encoder arguments
    cnn = parser.add_argument_group('CNN options')
    cnn.add_argument('--cnn-num-layers', type=int, default=6,
                        help='number convolutions')
    cnn.add_argument('--kernel-size', type=int, default=5,
                        help='size of convolution kernel')
    cnn.add_argument('--cnn-dropout', type=float, default=0.3,
                        help='dropout used in cnn')

    # Transformer encoder arguments
    trans = parser.add_argument_group('Transformer options')
    trans.add_argument('--N', type=int, default=6,
                        help='transformer options')
    trans.add_argument('--d-model', type=int, default=512,
                        help='transformer options')
    trans.add_argument('--d-ff', type=int, default=2048,
                        help='transformer options')
    trans.add_argument('--h', type=int, default=8,
                        help='transformer options')
    trans.add_argument('--trans-dropout', type=float, default=0.1,
                        help='dropout used in transformer')

    # Biaffine transformations
    biaff = parser.add_argument_group('Biaffine classifier arguments')
    biaff.add_argument('--mlp-arc-hidden', type=int, default=500,
                        help='number of hidden units in arc MLP')
    biaff.add_argument('--mlp-lab-hidden', type=int, default=100,
                        help='number of hidden units in label MLP')
    biaff.add_argument('--mlp-dropout', type=float, default=0.3,
                        help='dropout used in mlps')

    # Training.
    training = parser.add_argument_group('Training arguments')
    training.add_argument('--multi-gpu', action='store_true',
                        help='enable training on multiple GPUs')
    training.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate')
    training.add_argument('--epochs', type=int, default=10,
                        help='number of epochs of training')
    training.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    training.add_argument('--seed', type=int, default=42,
                        help='random seed')
    training.add_argument('--disable-cuda', action='store_true',
                        help='disable cuda')
    training.add_argument('--print-every', type=int, default=100,
                        help='report interval')
    training.add_argument('--plot-every', type=int, default=100,
                        help='plot interval')
    training.add_argument('--logdir', type=str,  default='log',
                        help='directory to log losses')
    training.add_argument('--checkpoints', type=str,  default='checkpoints/model.pt',
                        help='path to save the final model')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    if args.mode == 'predict':
        exit('Prediction not implemented yet...')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse

import train

def main():

    parser = argparse.ArgumentParser(description='Biaffine graph-based dependency parser')

    parser.add_argument('mode', type=str, choices=['train', 'predict'])
    # Data.
    parser.add_argument('--data', type=str, default='../../stanford-ptb',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default='vocab/train',
                        help='location of the preprocessed vocabulary')
    parser.add_argument('--char', action='store_true',
                        help='use character level word embeddings')
    # Model.
    parser.add_argument('--word_emb_dim', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--tag_emb_dim', type=int, default=20,
                        help='size of tag embeddings')
    parser.add_argument('--rnn_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM',
                        help='number of hidden units in RNN')
    parser.add_argument('--rnn_hidden', type=int, default=200,
                        help='number of hidden units in RNN')
    parser.add_argument('--rnn_num_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--mlp_arc_hidden', type=int, default=500,
                        help='number of hidden units in arc MLP')
    parser.add_argument('--mlp_lab_hidden', type=int, default=100,
                        help='number of hidden units in label MLP')
    # Training.
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.33,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--print_every', type=int, default=10,
                        help='report interval')
    parser.add_argument('--plot_every', type=int, default=100,
                        help='plot interval')
    parser.add_argument('--save', type=str,  default='models/model.pt',
                        help='path to save the final model')
    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)

if __name__ == '__main__':
    main()

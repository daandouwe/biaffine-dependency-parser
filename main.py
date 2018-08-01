#!/usr/bin/env python
import argparse

import train

def main():

    parser = argparse.ArgumentParser(description='Biaffine graph-based dependency parser')

    parser.add_argument('mode', type=str, choices=['train', 'predict'])
    # Data
    parser.add_argument('--data', type=str, default='../../stanford-ptb',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default='vocab/train',
                        help='location of the preprocessed vocabulary')
    parser.add_argument('--length_ordered', action='store_true',
                        help='order sentences by length so batches have minimal padding')
    # Embedding
    parser.add_argument('--char', action='store_true',
                        help='use character level word embeddings')
    parser.add_argument('--disable_tags', action='store_false',
                        help='do not use tags as additional input')
    parser.add_argument('--word_emb_dim', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--tag_emb_dim', type=int, default=50,
                        help='size of tag embeddings')
    parser.add_argument('--emb_dropout', type=float, default=0.3,
                        help='dropout used on embeddings')
    # Encoder
    parser.add_argument('--encoder', type=str, choices=['rnn', 'transformer'], default='rnn',
                        help='type of sentence encoder used')
    # RNN encoder arguments
    parser.add_argument('--rnn_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM',
                        help='number of hidden units in RNN')
    parser.add_argument('--rnn_hidden', type=int, default=400,
                        help='number of hidden units in RNN')
    parser.add_argument('--rnn_num_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--batch_first', type=bool, default=True,
                        help='number of layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.3,
                        help='dropout used in rnn')
    # Transformer encoder arguments
    parser.add_argument('--N', type=int, default=6,
                        help='')
    parser.add_argument('--d_model', type=int, default=512,
                        help='')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='')
    parser.add_argument('--h', type=int, default=8,
                        help='')
    parser.add_argument('--transformer_dropout', type=float, default=0.1,
                        help='dropout used in transformer')
    # Biaffine transformations
    parser.add_argument('--mlp_arc_hidden', type=int, default=500,
                        help='number of hidden units in arc MLP')
    parser.add_argument('--mlp_lab_hidden', type=int, default=100,
                        help='number of hidden units in label MLP')
    parser.add_argument('--mlp_dropout', type=float, default=0.3,
                        help='dropout used in mlps')
    # Training.
    parser.add_argument('--multi_gpu', action='store_true',
                        help='enable training on multiple GPUs')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--print_every', type=int, default=100,
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

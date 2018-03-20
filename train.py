import argparse
import os
import csv
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from data import Corpus
from model import BiAffineParser
from util import Timer, plot

######################################################################
# Parse command line arguments.
######################################################################
parser = argparse.ArgumentParser(description='Biaffine dependency parser')
parser.add_argument('--data', type=str, default='../../stanford-ptb',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='vocab/train',
                    help='location of the preprocessed vocabulary')
parser.add_argument('--word_emb_dim', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--tag_emb_dim', type=int, default=20,
                    help='size of tag embeddings')
parser.add_argument('--lstm_hidden', type=int, default=200,
                    help='number of hidden units in LSTM')
parser.add_argument('--lstm_num_layers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--mlp_arc_hidden', type=int, default=100,
                    help='number of hidden units in arc MLP')
parser.add_argument('--mlp_lab_hidden', type=int, default=100,
                    help='number of hidden units in label MLP')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
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

np.random.seed(args.seed)
torch.manual_seed(args.seed)

######################################################################
# Initialize the data, model, and optimizer.
######################################################################
corpus = Corpus(data_path=args.data, vocab_path=args.vocab)
model = BiAffineParser(word_vocab_size=len(corpus.dictionary.w2i),
                       word_emb_dim=args.word_emb_dim,
                       tag_vocab_size=len(corpus.dictionary.t2i),
                       tag_emb_dim=args.tag_emb_dim,
                       emb_dropout=args.dropout,
                       lstm_hidden=args.lstm_hidden,
                       lstm_num_layers=args.lstm_num_layers,
                       lstm_dropout=args.dropout,
                       mlp_arc_hidden=args.mlp_arc_hidden,
                       mlp_lab_hidden=args.mlp_lab_hidden,
                       mlp_dropout=args.dropout,
                       num_labels=len(corpus.dictionary.l2i),
                       criterion=nn.CrossEntropyLoss())
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

######################################################################
# Useful functions.
######################################################################
def save(model, path=args.save):
    torch.save(model, path)

def write(loss, train_acc, val_acc, path='csv'):
    """
    Write out the loss and accuracy to CSV files.
    """
    with open(os.path.join(path, 'loss.train.csv'), 'w') as f:
        writer = csv.writer(f)
        names = [["loss"]]
        logs = [[l] for l in loss]
        writer.writerows(names + logs)
    with open(os.path.join(path, 'acc.train.csv'), 'w') as f:
        writer = csv.writer(f)
        names = [["train_arc_acc", "train_lab_acc"]]
        writer.writerows(names + train_acc)
    with open(os.path.join(path, 'acc.val.csv'), 'w') as f:
        writer = csv.writer(f)
        names = [["val_arc_acc", "val_lab_acc"]]
        writer.writerows(names + val_acc)

def arc_accuracy(S_arc, heads, eps=1e-10):
    """
    Accuracy of the arc predictions.
    """
    _, pred = S_arc.max(dim=-2)
    mask = (heads > 0).float()
    accuracy = torch.sum((pred == heads).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data.numpy()[0]

def lab_accuracy(S_lab, heads, labels, eps=1e-10):
    """
    Accuracy of label predictions on the gold arcs.
    """
    _, pred = S_lab.max(dim=1)
    pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
    mask = (heads > 0).float()
    accuracy = torch.sum((pred == labels).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data.numpy()[0]

def evaluate(model, corpus):
    """
    Evaluate the arc and label accuracy of the model on the development corpus.
    """
    # Turn on evaluation mode to disable dropout.
    model.eval()
    dev_batches = corpus.dev.batches(batch_size, length_ordered=True)
    arc_acc, lab_acc = 0, 0
    for k, batch in enumerate(dev_batches, 1):
        words, tags, heads, labels = batch
        S_arc, S_lab = model(words, tags)
        arc_acc += arc_accuracy(S_arc, heads)
        lab_acc += lab_accuracy(S_lab, heads, labels)
    arc_acc /= k
    lab_acc /= k
    return arc_acc, lab_acc

######################################################################
# The training step.
######################################################################
def train(batch, alpha=1.):
    """
    Performs one forward pass and parameter update.
    """
    # Forward pass.
    words, tags, heads, labels = batch
    S_arc, S_lab = model(words, tags)
    # Compute loss.
    arc_loss = model.arc_loss(S_arc, heads)
    lab_loss = model.lab_loss(S_lab, heads, labels)
    loss = arc_loss + alpha*lab_loss

    # Update parameters.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Convert loss to numpy float for printing.
    loss = loss.data.numpy()[0]

    return S_arc, S_lab, loss

######################################################################
# Train!
######################################################################
timer = Timer()
n_batches = len(corpus.train.words) // args.batch_size
train_loss, train_acc, val_acc, test_acc = [], [], [], []
best_val_acc, best_epoch = 0, 0
fig, ax = plt.subplots()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # Turn on dropout.
        model.train()
        # Get a new set of shuffled training batches.
        train_batches = corpus.train.batches(args.batch_size, length_ordered=False)
        for step, batch in enumerate(train_batches, 1):
            words, tags, heads, labels = batch
            S_arc, S_lab, loss = train(batch)
            train_loss.append(loss)
            if step % args.print_every == 0:
                arc_train_acc = arc_accuracy(S_arc, heads)
                lab_train_acc = lab_accuracy(S_lab, heads, labels)
                train_acc.append([arc_train_acc, lab_train_acc])
                print('Epoch {} | Step {}/{} | Avg loss {:.4f} | Arc accuracy {:.2f}% | '
                      'Label accuracy {:.2f}% | {:.0f} sents/sec |'
                        ''.format(epoch, step, n_batches, np.mean(train_loss[-args.print_every:]),
                        100*arc_train_acc, 100*lab_train_acc,
                        args.batch_size*args.print_every/timer.elapsed()), end='\r')
            if step % args.plot_every == 0:
                plot(corpus, model, fig, ax, step)
        # Evaluate model on validation set.
        arc_val_acc, lab_val_acc = evaluate(model, corpus)
        val_acc.append([arc_val_acc, lab_val_acc])
        # Save model if it is the best so far.
        if arc_val_acc > best_val_acc:
            save(model)
            best_val_acc = arc_val_acc
            best_epoch = epoch
        write(train_loss, train_acc, val_acc)
        # End epoch with some useful info in the terminal.
        print('-' * 89)
        print('| End of epoch {} | Time: {:.2f}s | Valid accuracy {:.2f}% |'
                ' Best accuracy {:.2f}% (epoch {})'.format(epoch,
                (time.time() - epoch_start_time), 100*arc_val_acc, 100*best_val_acc, best_epoch))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

######################################################################
# Wrap-up.
######################################################################
write(train_loss, train_acc, val_acc)
arc_val_acc, lab_val_acc = evaluate(model, corpus)
if arc_val_acc > best_val_acc:
    save(model)

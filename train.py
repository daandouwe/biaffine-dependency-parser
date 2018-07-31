import os
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from data import Corpus
from model import BiAffineParser
from util import Timer, write

######################################################################
# Useful functions.
######################################################################
def arc_accuracy(S_arc, heads, eps=1e-10):
    """Accuracy of the arc predictions based on gready head prediction."""
    _, pred = S_arc.max(dim=-2)
    mask = (heads > 0).float()
    accuracy = torch.sum((pred == heads).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data.numpy()[0]

def lab_accuracy(S_lab, heads, labels, eps=1e-10):
    """Accuracy of label predictions on the gold arcs."""
    _, pred = S_lab.max(dim=1)
    pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
    mask = (heads > 0).float()
    accuracy = torch.sum((pred == labels).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data.numpy()[0]

def evaluate(model, corpus):
    """Evaluate the arc and label accuracy of the model on the development corpus."""
    # Turn on evaluation mode to disable dropout.
    model.eval()
    dev_batches = corpus.dev.batches(32, length_ordered=True)
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
def train(model, batch, optimizer):
    """Performs one forward pass and parameter update."""
    # Forward pass.
    words, tags, heads, labels = batch
    S_arc, S_lab = model(words, tags)
    # Compute loss.
    arc_loss = model.arc_loss(S_arc, heads)
    lab_loss = model.lab_loss(S_lab, heads, labels)
    loss = arc_loss + lab_loss

    # Update parameters.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Convert losses to numpy float for printing.
    loss = loss.data.numpy()[0]

    return S_arc, S_lab, loss

######################################################################
# Train!
######################################################################
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize the data, model, and optimizer.
    corpus = Corpus(data_path=args.data, vocab_path=args.vocab, char=args.char)
    model = BiAffineParser(word_vocab_size=len(corpus.dictionary.w2i),
                           word_emb_dim=args.word_emb_dim,
                           tag_vocab_size=len(corpus.dictionary.t2i),
                           tag_emb_dim=args.tag_emb_dim,
                           emb_dropout=args.dropout,
                           rnn_type=args.rnn_type,
                           rnn_hidden=args.rnn_hidden,
                           rnn_num_layers=args.rnn_num_layers,
                           rnn_dropout=args.dropout,
                           mlp_arc_hidden=args.mlp_arc_hidden,
                           mlp_lab_hidden=args.mlp_lab_hidden,
                           mlp_dropout=args.dropout,
                           num_labels=len(corpus.dictionary.l2i),
                           criterion=nn.CrossEntropyLoss(),
                           char=args.char)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('Start of training..')
    timer = Timer()
    n_batches = len(corpus.train.words) // args.batch_size
    train_loss, train_acc, val_acc, test_acc = [], [], [], []
    best_val_acc, best_epoch = 0, 0
    # fig, ax = plt.subplots()
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            # Turn on dropout.
            model.train()
            # Get a new set of shuffled training batches.
            train_batches = corpus.train.batches(args.batch_size, length_ordered=False)
            for step, batch in enumerate(train_batches, 1):
                words, tags, heads, labels = batch
                S_arc, S_lab, loss = train(model, batch, optimizer)
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
                # if step % args.plot_every == 0:
                    # plot(corpus, model, fig, ax, step)
            # Evaluate model on validation set.
            arc_val_acc, lab_val_acc = evaluate(model, corpus)
            val_acc.append([arc_val_acc, lab_val_acc])
            # Save model if it is the best so far.
            if arc_val_acc > best_val_acc:
                torch.save(model, args.save)
                best_val_acc = arc_val_acc
                best_epoch = epoch
            write(train_loss, train_acc, val_acc)
            # End epoch with some useful info in the terminal.
            print('-' * 89)
            print('| End of epoch {} | Time elapsed: {:.2f}s | Valid accuracy {:.2f}% |'
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
        torch.save(model, args.save)

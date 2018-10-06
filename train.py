import os
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from model import PAD_INDEX
from data import Corpus
from model import make_model
from optimizer import get_std_transformer_opt
from util import Timer, write_losses


LOSSES = dict(train_loss=[], train_acc=[], val_acc=[], test_acc=[])


def arc_accuracy(S_arc, heads, eps=1e-10):
    """Accuracy of the arc predictions based on gready head prediction."""
    _, pred = S_arc.max(dim=-2)
    mask = (heads != PAD_INDEX).float()
    accuracy = torch.sum((pred == heads).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data[0]


def lab_accuracy(S_lab, heads, labels, eps=1e-10):
    """Accuracy of label predictions on the gold arcs."""
    _, pred = S_lab.max(dim=1)
    pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
    mask = (heads != PAD_INDEX).float()
    accuracy = torch.sum((pred == labels).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    return torch.mean(accuracy).data[0]


def evaluate(args, model, corpus):
    """Evaluate the arc and label accuracy of the model on the development corpus."""
    # Turn on evaluation mode to disable dropout.
    model.eval()
    dev_batches = corpus.dev.batches(256, length_ordered=True)
    arc_acc, lab_acc = 0, 0
    for k, batch in enumerate(dev_batches, 1):
        words, tags, heads, labels = batch
        if args.cuda:
            words, tags, heads, labels = words.cuda(), tags.cuda(), heads.cuda(), labels.cuda()
        S_arc, S_lab = model(words=words, tags=tags)
        arc_acc += arc_accuracy(S_arc, heads)
        lab_acc += lab_accuracy(S_lab, heads, labels)
    arc_acc /= k
    lab_acc /= k
    return arc_acc, lab_acc


class SimpleLossCompute:
    """A simple loss compute and train function on one device."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def __call__(self, words, tags, heads, labels):
        # Forward pass.
        S_arc, S_lab = self.model(words=words, tags=tags)
        # Compute loss.
        arc_loss = self.model.arc_loss(S_arc, heads)
        lab_loss = self.model.lab_loss(S_lab, heads, labels)
        loss = arc_loss + lab_loss
        # Update parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = dict(loss=loss.data[0], arc_loss=arc_loss.data[0], lab_loss=lab_loss.data[0])
        return S_arc, S_lab, loss_dict


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function.

    Only difference with SimpleLossCompute is we need to access loss
    through model.module.
    """
    def __init__(self, model, optimizer, devices, output_device=None):
        self.model = model
        self.optimizer = optimizer
        self.devices = devices
        self.output_device = output_device if output_device is not None else devices[0]

    def __call__(self, words, tags, heads, labels):
        # Forward pass.
        S_arc, S_lab = self.model(words=words, tags=tags)
        # Compute loss.
        arc_loss = self.model.module.arc_loss(S_arc, heads)
        lab_loss = self.model.module.lab_loss(S_lab, heads, labels)
        loss = arc_loss + lab_loss
        # Update parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = dict(loss=loss.data[0], arc_loss=arc_loss.data[0], lab_loss=lab_loss.data[0])
        return S_arc, S_lab, loss_dict


def run_epoch(args, model, corpus, train_step):
    model.train()
    nbatches = len(corpus.train.words) // args.batch_size
    start_time = time.time()
    # Get a new set of shuffled training batches.
    train_batches = corpus.train.batches(args.batch_size, length_ordered=args.disable_length_ordered)
    ntokens = 0
    for step, batch in enumerate(train_batches, 1):
        words, tags, heads, labels = batch
        if args.cuda:
            words, tags, heads, labels = words.cuda(), tags.cuda(), heads.cuda(), labels.cuda()
        S_arc, S_lab, loss_dict = train_step(words, tags, heads, labels)
        ntokens += words.size(0) * words.size(1)
        LOSSES['train_loss'].append(loss_dict['loss'])
        if step % args.print_every == 0:
            arc_train_acc = arc_accuracy(S_arc, heads)
            lab_train_acc = lab_accuracy(S_lab, heads, labels)
            LOSSES['train_acc'].append([arc_train_acc, lab_train_acc])
            print(
                '| Step {:5d}/{:5d} ({:.0f}%)| Avg loss {:3.4f} | Arc acc {:4.2f}% '
                '| Label acc {:4.2f}% | {:4.0f} tokens/sec |'.format(
                    step,
                    nbatches,
                    100*step/nbatches,
                    np.mean(LOSSES['train_loss'][-args.print_every:]),
                    100*arc_train_acc,
                    100*lab_train_acc,
                    ntokens/(time.time() - start_time)),
            )


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    print('Using cuda: {}'.format(args.cuda))

    # Initialize the data, model, and optimizer.
    corpus = Corpus(data_path=args.data, vocab_path=args.vocab, char=args.use_chars)
    model = make_model(
                args,
                word_vocab_size=len(corpus.dictionary.w2i),
                tag_vocab_size=len(corpus.dictionary.t2i),
                num_labels=len(corpus.dictionary.l2i)
            )
    print('Embedding parameters: {:,}'.format(model.embedding.num_parameters))
    print('Encoder parameters: {:,}'.format(model.encoder.num_parameters))
    print('Total model parameters: {:,}'.format(model.num_parameters))
    if args.cuda:
        model.cuda()

    if args.encoder == 'transformer':
        optimizer = get_std_transformer_opt(args, model)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        device_count = torch.cuda.device_count()
        if args.multi_gpu:
            devices = list(range(device_count))
            model = nn.DataParallel(model, device_ids=devices)
            train_step = MultiGPULossCompute(model, optimizer, devices)
            print('Training on {} GPUs: {}.'.format(device_count, devices))
        else:
            train_step = SimpleLossCompute(model, optimizer)
            print('Training on 1 device out of {} availlable.'.format(device_count))
    else:
        train_step = SimpleLossCompute(model, optimizer)

    timer = Timer()
    best_val_acc = 0.
    best_epoch = 0
    print('Start of training..')
    try:
        for epoch in range(1, args.epochs+1):
            run_epoch(args, model, corpus, train_step)

            # Evaluate model on validation set.
            # TODO: replace this with a UAS and LAS eval instead of this proxy
            arc_val_acc, lab_val_acc = evaluate(args, model, corpus)
            LOSSES['val_acc'].append([arc_val_acc, lab_val_acc])

            # Save model if it is the best so far.
            if arc_val_acc > best_val_acc:
                torch.save(model, args.checkpoints)
                best_val_acc = arc_val_acc
                best_epoch = epoch

            write_losses(LOSSES['train_loss'], LOSSES['train_acc'], LOSSES['val_acc'], args.logdir)
            # End epoch with some useful info in the terminal.
            print('-' * 89)
            print(
                '| End of epoch {:3d}/{} | Time {:5.2f}s | Valid accuracy {:3.2f}% |'
                ' Best accuracy {:3.2f}% (epoch {:3d}) |'.format(
                    epoch,
                    args.epochs,
                    timer.elapsed(),
                    100*arc_val_acc,
                    100*best_val_acc,
                    best_epoch)
            )
            print('-' * 89)
    except KeyboardInterrupt:
        print()
        print('-' * 89)
        print('Exiting from training early')

    write_losses(LOSSES['train_loss'], LOSSES['train_acc'], LOSSES['val_acc'], args.logdir)
    arc_val_acc, lab_val_acc = evaluate(args, model, corpus)
    if arc_val_acc > best_val_acc:
        torch.save(model, args.checkpoints)
        best_val_acc = arc_val_acc
        best_epoch = epoch

    print('=' * 89)
    print('| End of training | Best validation accuracy {:3.2f} (epoch {}) |'.format(
        100*best_val_acc, best_epoch))
    print('=' * 89)

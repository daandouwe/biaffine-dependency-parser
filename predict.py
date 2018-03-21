import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import Dictionary, Corpus, PAD_INDEX
from mst import mst

def plot(S_arc, heads):
    fig, ax = plt.subplots()
    # Make a 0/1 gold adjacency matrix.
    n = heads.size(1)
    G = np.zeros((n, n))
    heads = heads.squeeze().data.numpy()
    G[heads, np.arange(n)] = 1.
    im = ax.imshow(G, vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig('img/gold.pdf')
    plt.cla()
    # Plot the predicted adjacency matrix
    A = F.softmax(S_arc.squeeze(0), dim=0)
    fig, ax = plt.subplots()
    im = ax.imshow(A.data.numpy(), vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig('img/a.pdf')
    plt.cla()
    plt.clf()

def predict(model, words, tags):
    """
    :param words: a python list with word indices.
    :param tags: a python list with tag indices.
    """
    if type(words) == type(tags) == list:
        # Convert the lists into input for the PyTorch model.
        words = Variable(torch.LongTensor([words]))
        tags = Variable(torch.LongTensor([tags]))
    # Dissable dropout.
    model.eval()
    # Predict arc and label score matrices.
    S_arc, S_lab = model(words, tags)

    # Predict heads
    S = S_arc[0].data.numpy()
    heads = mst(S)

    # Predict labels
    S_lab = S_lab[0]
    select = torch.LongTensor(heads).unsqueeze(0).expand(S_lab.size(0), -1)
    select = Variable(select)
    selected = torch.gather(S_lab, 1, select.unsqueeze(1)).squeeze(1)
    _, labels = selected.max(dim=0)
    labels = labels.data.numpy()
    return heads, labels

if __name__ == '__main__':

    data_path = '../../stanford-ptb'
    vocab_path = 'vocab/train'
    model_path = 'models/model.pt'

    dictionary = Dictionary(vocab_path)
    corpus = Corpus(data_path=data_path, vocab_path=vocab_path)
    model = torch.load(model_path)
    batches = corpus.train.batches(1)

    words, tags, heads, labels = next(batches)
    S_arc, S_lab = model(words, tags)

    plot(S_arc, heads)
    words = tags = [1, 2, 3, 4]
    heads_pred, labels_pred = predict(model, words, tags)
    print(heads_pred, '\n', heads[0].data.numpy())
    print(labels_pred, '\n', labels[0].data.numpy())

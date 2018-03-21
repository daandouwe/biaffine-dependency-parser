import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import Dictionary, Corpus, PAD_INDEX
from predict import predict, predict_batch

class CONLL:

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.words = []
        self.tags = []
        self.heads = []
        self.labels = []
        self.lengths = []

    def add(self, words, tags, heads, labels):
        self.words.append([self.dictionary.i2w[i] for i in words])
        self.tags.append([self.dictionary.i2t[i] for i in tags])
        self.heads.append(heads)
        self.labels.append([self.dictionary.i2l[i] for i in labels])

    def write(self, path='predicted.conll'):
        """"
        Write the data out as a conll file (Stanford style).
        """
        with open(path, 'w') as f:
            for line in zip(self.words, self.tags, self.heads, self.labels):
                words, tags, heads, labels = line
                lines = zip(words[1:], tags[1:], heads[1:], labels[1:])
                for i, (w, t, h, l) in enumerate(lines, 1):
                    print(i, w, '_', t, t, '_', h, l, '_', '_', sep='\t', file=f)
                print(file=f)

class Parser:
    def __init__(self, corpus, model):
        pass
        self.model = model
        self.corpus = corpus

    def batch_eval(self, batch_size=128):
        conll = CONLL(corpus.dictionary)
        batches = self.corpus.dev.batches(batch_size, shuffle=False)
        self.model.eval()
        for i, batch in enumerate(batches):
            print('Batch:', i, end='\r')
            words, tags, heads, labels = batch
            # Predict score matrices for the batch.
            S_arc, S_lab = self.model(words, tags)
            for i in range(words.size(0)):
                # Find the sentence length.
                n = (words[i] != PAD_INDEX).int().sum().data.numpy()[0]
                # Predict for the selected parts that are the sentence.
                heads_pred, labels_pred = predict_batch(S_arc[i, :n, :n],
                                                        S_lab[i, :, :n, :n],
                                                        tags[i, :n])
                conll.add(words[i].data.numpy(), tags[i].data.numpy(),
                            heads_pred, labels_pred)
        return conll

    def eval(self, corpus):
        self.model.eval()
        batches = self.corpus.dev.batches(1, shuffle=False)
        conll = CONLL(corpus.dictionary)
        for i, batch in enumerate(batches):
            print('Batch:', i, end='\r')
            words, tags, heads, labels = batch
            heads_pred, labels_pred = predict(model, words, tags)
            words = words[0].data.numpy()
            tags = tags[0].data.numpy()
            conll.add(words, tags, heads_pred, labels_pred)
        return conll

if __name__ == '__main__':

    data_path = '../../stanford-ptb'
    vocab_path = 'vocab/train'
    model_path = 'models/trained/model.pt'

    gold_path = '../../stanford-ptb/dev-stanford-raw.conll'
    predict_path = 'predicted.conll'
    result_path = 'result.txt'

    corpus = Corpus(data_path=data_path, vocab_path=vocab_path)
    model = torch.load(model_path)

    parser = Parser(corpus, model)
    conll = parser.batch_eval()

    # Write the conll as text.
    conll.write(predict_path)
    # Evaluate the predicted conll.
    os.system('perl eval.pl -g {0} -s {1} > {2}'.format(gold_path, predict_path, result_path))

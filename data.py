import os
from collections import defaultdict
import numpy as np

import torch
from torch.autograd import Variable

PAD_TOKEN = '<pad>'
PAD_TAG = 'PAD'
PAD_LABEL = '_pad_'
PAD_INDEX = 0

UNK_TOKEN = '<unk>'
UNK_TAG = 'UNK'
UNK_LABEL = '_unk_'
UNK_INDEX = 1

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '_root_'
ROOT_INDEX = 2

def pad(batch):
    """
    Pad a batch of irregular length indices, and return as
    a Variable so it is ready as input for a PyTorch model.
    """
    lens = list(map(len, batch))
    max_len = max(lens)
    padded_batch = []
    for k, seq in zip(lens, batch):
        padded =  seq + (max_len - k)*[PAD_INDEX]
        padded_batch.append(padded)
    return Variable(torch.LongTensor(padded_batch))

class Dictionary:
    """
    A dependency parse dictionary.
    """
    def __init__(self, path):
        self.w2i = defaultdict(lambda: UNK_INDEX)
        self.t2i = defaultdict(lambda: UNK_INDEX)
        self.l2i = defaultdict(lambda: UNK_INDEX)

        self.i2w = defaultdict(lambda: UNK_TOKEN)
        self.i2t = defaultdict(lambda: UNK_TAG)
        self.i2l = defaultdict(lambda: UNK_LABEL)

        self.add_word(PAD_TOKEN)
        self.add_word(UNK_TOKEN)
        self.add_word(ROOT_TOKEN)

        self.add_tag(PAD_TAG)
        self.add_tag(UNK_TAG)
        self.add_tag(ROOT_TAG)

        self.add_label(PAD_LABEL)
        self.add_label(UNK_LABEL)
        self.add_label(ROOT_LABEL)

        self.read(path)

    def add_word(self, word, processed_word=None, unk=False):
        if word not in self.w2i:
            if unk:
                self.i2w[UNK_INDEX] = UNK_TOKEN
                self.w2i[word] = UNK_INDEX
            else:
                i = len(self.i2w)
                self.i2w[i] = word
                self.w2i[word] = i

    def add_tag(self, tag):
        if tag not in self.t2i:
            i = len(self.i2t)
            self.i2t[i] = tag
            self.t2i[tag] = i

    def add_label(self, label):
        if label not in self.l2i:
            i = len(self.i2l)
            self.i2l[i] = label
            self.l2i[label] = i

    def read(self, path):
        with open(path + ".words.txt", 'r') as f:
            for line in f:
                word, processed_word, _ = line.split()
                unk = bool(word != processed_word)
                self.add_word(word, processed_word, unk=unk)
        with open(path + ".tags.txt", 'r') as f:
            for line in f:
                tag, _ = line.split()
                self.add_tag(tag)
        with open(path + ".labels.txt", 'r') as f:
            for line in f:
                label, _ = line.split()
                self.add_label(label)

class Data:
    """
    A dependency parse dataset.
    """
    def __init__(self, path, dictionary):
        self.words = []
        self.tags = []
        self.heads = []
        self.labels = []
        self.lengths = []
        self.read(path, dictionary)

    def read(self, path, dictionary):
        with open(path, 'r') as f:
            ws, ts, hs, ls, n = self.newline()
            for line in f:
                fields = line.split()
                if fields:
                    w, t, h, l = fields[1], fields[3], fields[6], fields[7]
                    ws.append(dictionary.w2i[w.lower()])
                    ts.append(dictionary.t2i[t])
                    hs.append(int(h))
                    ls.append(dictionary.l2i[l])
                    n += 1
                else:
                    self.words.append(ws)
                    self.tags.append(ts)
                    self.heads.append(hs)
                    self.labels.append(ls)
                    self.lengths.append(n)
                    ws, ts, hs, ls, n = self.newline()

    def newline(self):
        """
        Each sentence in our data-set must start with these indices.
        Note the convention: the root has itelf as head.
        """
        return [ROOT_INDEX], [ROOT_INDEX], [0], [ROOT_INDEX], 1

    def order(self):
        old_order = zip(range(len(self.lengths)), self.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self.words = [self.words[i] for i in new_order]
        self.tags = [self.tags[i] for i in new_order]
        self.heads = [self.heads[i] for i in new_order]
        self.labels = [self.labels[i] for i in new_order]
        self.lengths = [self.lengths[i] for i in new_order]

    def shuffle_blocks(self):
        """
        TODO: Shuffle the blocks of same-size sentences.
        """
        pass

    def shuffle(self):
        n = len(self.words)
        new_order = list(range(0, n))
        np.random.shuffle(new_order)
        self.words = [self.words[i] for i in new_order]
        self.tags = [self.tags[i] for i in new_order]
        self.heads = [self.heads[i] for i in new_order]
        self.labels = [self.labels[i] for i in new_order]
        self.lengths = [self.lengths[i] for i in new_order]

    def batches(self, batch_size, shuffle=True, length_ordered=False):
        """
        An iterator over batches.
        """
        n = len(self.words)
        batch_order = list(range(0, n, batch_size))
        if shuffle:
            self.shuffle()
            np.random.shuffle(batch_order)
        if length_ordered:
            self.order()
        for i in batch_order:
            words = pad(self.words[i:i+batch_size])
            tags = pad(self.tags[i:i+batch_size])
            heads = pad(self.heads[i:i+batch_size])
            labels = pad(self.labels[i:i+batch_size])
            yield words, tags, heads, labels

    def batches_list(self, batch_size, length_ordered=False):
        """
        The batches as list.
        """
        if length_ordered:
            self.order()
        else:
            self.shuffle()
        n = len(self.words)
        batch_order = list(range(0, n, batch_size))
        np.random.shuffle(batch_order)
        batches = []
        for i in batch_order:
            words = pad(self.words[i:i+batch_size])
            tags = pad(self.tags[i:i+batch_size])
            heads = pad(self.heads[i:i+batch_size])
            labels = pad(self.labels[i:i+batch_size])
            batches.append((words, tags, heads, labels))
        return batches

class Corpus:
    """
    A corpus of three datasets (train, development, and test) and a dictionary.
    """
    def __init__(self, vocab_path="vocab/train", data_path="stanford-ptb"):
        self.dictionary = Dictionary(vocab_path)
        self.train = Data(os.path.join(data_path, "train-stanford-raw.conll"), self.dictionary)
        self.dev = Data(os.path.join(data_path, "dev-stanford-raw.conll"), self.dictionary)
        self.test = Data(os.path.join(data_path, "test-stanford-raw.conll"), self.dictionary)


if __name__ == "__main__":
    # Example usage:
    corpus = Corpus(data_path="../../stanford-ptb")
    batches = corpus.train.batches(16)
    for _ in range(10):
        words, tags, heads, labels = next(batches)

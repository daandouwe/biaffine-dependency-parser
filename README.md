# biaffine-dependency-parser

A PyTorch implementation of the neural dependency parser described in [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

# TODO
- [x] Add MST algorithm for decoding.
- [x] Write predicted parses to conll file.
- [ ] Label loss converges very fast, which seems to hurt the arc accuracy.
- [ ] A couple of full runs of the model for results.
- [ ] Perfom some ablation experiments.
- [ ] Make it CUDA.
- [ ] Work on character-level embedding of words (CNN or LSTM).
- [ ] Make a version that works without input POS-tags at prediction time. See [SpaCy's parser](https://spacy.io/api/) and the paper that it is based on: [Stack-propagation: Improved Representation Learning for Syntax](https://www.semanticscholar.org/paper/Stack-propagation%3A-Improved-Representation-Learning-Zhang-Weiss/0c133f79b23e8c680891d2e49a66f0e3d37f1466).
- [ ] Try with different RNN cell (GRU, RAN).
- [ ] Try with CNN instead of LSTM for context embeddings. (e.g. see spaCy's parser [here]( or [this paper](https://arxiv.org/pdf/1803.01271.pdf).

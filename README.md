# biaffine-dependency-parser

A PyTorch implementation of the neural dependency parser described in [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

# TODO
- [x] Add MST algorithm for decoding.
- [x] Write predicted parses to conll file.
- [ ] Label loss converges very fast, which seems to hurt the arc accuracy.
- [ ] A couple of full runs of the model for results.
- [ ] Perfom some ablation experiments.
- [ ] Make it CUDA.
- [x] Work on character-level embedding of words (CNN or LSTM).
- [ ] Make a version that works without input POS-tags at prediction time. See [spaCy's parser](https://spacy.io/api/) and these papers that it is based on: [Stack-propagation: Improved Representation Learning for Syntax](https://www.semanticscholar.org/paper/Stack-propagation%3A-Improved-Representation-Learning-Zhang-Weiss/0c133f79b23e8c680891d2e49a66f0e3d37f1466) and [Deep multi-task learning with low level tasks supervised at lower layers](https://pdfs.semanticscholar.org/03ad/06583c9721855ccd82c3d969a01360218d86.pdf?_ga=2.12476148.1950369760.1522163668-1479393485.1519147866).
- [ ] Try with different RNN cell (GRU, RAN).
- [ ] Try with CNN instead of LSTM for context embeddings. (again see spaCy's parser).
- [ ] Implement transformer as encoder, then make training parallelizable.

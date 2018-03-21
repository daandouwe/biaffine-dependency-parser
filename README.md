# biaffine-dependency-parser

A PyTorch implementation of the neural dependency parser described in [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

# TODO
* Add MST algorithm for decoding.
* Write predicted parses to conll file.
* Label loss converges very fast, which seems to hurt the arc accuracy.
* A couple of full runs of the model for results.
* Perfom some ablation experiments.
* Make it CUDA.
* Work on character-level embedding of words (CNN or LSTM).
* Try with different RNN cell (GRU, RAN).
* Try with CNN instead of LSTM for context embeddings. 

# Biaffine dependency-parser

A PyTorch implementation of the neural dependency parser described in [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

# Usage
```
usage: main.py [-h] [--data DATA] [--vocab VOCAB] [--disable_length_ordered]
               [--use_glove] [--use_char]
               [--char_encoder {rnn,cnn,transformer}] [--disable_tags]
               [--word_emb_dim WORD_EMB_DIM] [--tag_emb_dim TAG_EMB_DIM]
               [--emb_dropout EMB_DROPOUT] [--encoder {rnn,cnn,transformer}]
               [--rnn_type {RNN,GRU,LSTM}] [--rnn_hidden RNN_HIDDEN]
               [--rnn_num_layers RNN_NUM_LAYERS] [--batch_first BATCH_FIRST]
               [--rnn_dropout RNN_DROPOUT] [--N N] [--d_model D_MODEL]
               [--d_ff D_FF] [--h H]
               [--transformer_dropout TRANSFORMER_DROPOUT]
               [--mlp_arc_hidden MLP_ARC_HIDDEN]
               [--mlp_lab_hidden MLP_LAB_HIDDEN] [--mlp_dropout MLP_DROPOUT]
               [--multi_gpu] [--lr LR] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--seed SEED] [--disable_cuda]
               [--print_every PRINT_EVERY] [--plot_every PLOT_EVERY]
               [--save SAVE]
               {train,predict}

Biaffine graph-based dependency parser

positional arguments:
  {train,predict}

optional arguments:
  -h, --help            show this help message and exit

Data:
  --data DATA           location of the data corpus
  --vocab VOCAB         location of the preprocessed vocabulary
  --disable_length_ordered
                        do not order sentences by length so batches have more
                        padding

Embedding options:
  --use_glove           use pretrained glove embeddings
  --use_char            use character level word embeddings
  --char_encoder {rnn,cnn,transformer}
                        type of character encoder used for word embeddings
  --disable_tags        do not use tags as additional input
  --word_emb_dim WORD_EMB_DIM
                        size of word embeddings
  --tag_emb_dim TAG_EMB_DIM
                        size of tag embeddings
  --emb_dropout EMB_DROPOUT
                        dropout used on embeddings

Encoder options:
  --encoder {rnn,cnn,transformer}
                        type of sentence encoder used
  --rnn_type {RNN,GRU,LSTM}
                        number of hidden units in RNN
  --rnn_hidden RNN_HIDDEN
                        number of hidden units in RNN
  --rnn_num_layers RNN_NUM_LAYERS
                        number of layers
  --batch_first BATCH_FIRST
                        number of layers
  --rnn_dropout RNN_DROPOUT
                        dropout used in rnn
  --N N                 transformer options
  --d_model D_MODEL     transformer options
  --d_ff D_FF           transformer options
  --h H                 transformer options
  --transformer_dropout TRANSFORMER_DROPOUT
                        dropout used in transformer

Biaffine classifier arguments:
  --mlp_arc_hidden MLP_ARC_HIDDEN
                        number of hidden units in arc MLP
  --mlp_lab_hidden MLP_LAB_HIDDEN
                        number of hidden units in label MLP
  --mlp_dropout MLP_DROPOUT
                        dropout used in mlps

Training arguments:
  --multi_gpu           enable training on multiple GPUs
  --lr LR               initial learning rate
  --epochs EPOCHS       number of epochs of training
  --batch_size BATCH_SIZE
                        batch size
  --seed SEED           random seed
  --disable_cuda        disable CUDA
  --print_every PRINT_EVERY
                        report interval
  --plot_every PLOT_EVERY
                        plot interval
  --save SAVE           path to save the final model
```

# TODO
- [x] Add MST algorithm for decoding.
- [x] Write predicted parses to conll file.
- [ ] Label loss converges very fast, which seems to hurt the arc accuracy.
- [x] A couple of full runs of the model for results.
- [ ] Perform some ablation experiments.
- [x] Make it CUDA.
- [x] Enable multi-gpu training
- [x] Work on character-level embedding of words (CNN or LSTM).
- [ ] Disable input POS-tags at prediction time but train with them using mutli-task learning. See [spaCy's parser](https://spacy.io/api/) and these papers that it is based on: [Stack-propagation: Improved Representation Learning for Syntax](https://arxiv.org/pdf/1603.06598.pdf) and [Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038).
- [x] Implement RNN options: RNN, GRU, (RAN?)
- [ ] Different encoder: CNN (again see [spaCy's parser](https://spacy.io/api/)).
- [x] Different encoder: [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).
- [ ] Load pretrained GloVe embeddings.
- [ ] Character level word embeddings: RNN
- [x] Character level word embeddings: CNN

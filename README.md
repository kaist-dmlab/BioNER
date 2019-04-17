# DTranNER

## Biomedical Named Entity Recognizer

**DTranNER** is a deep-learning-based method suited for biomedical named entity recognition that obtains state-of-the-art performance in NER on the four biomedical benchmark corpora (BC2GM, BC4CHEMD, BC5CDR, and NCBI-Diesease). **DTranNER** equips with label-label transition model to describe ever-changing relations between neighboring labels.

## Initial setup

To use **DTranNER**, you need to install Python 2.7, with Numpy, Spacy, and Theano 1.0.0.

## Usage
**To be updated**

## Model Training
For model training, we recommend using GPU.
```
./train_crf.py --character_model ['cnn' or 'lstm'] --lstm_hidden_dim [e.g. 800] --minibatch_size [e.g. 10] --gradient_threshold [e.g. 20] --dataset [target corpus] --cnn_case_sensitivity ['yes' or 'no'] --config_layer_normalization ['yes' or 'no'] --logging_label_prediction ['yes' or 'no']
```

## Download Word Embedding
We initialize the word embedding matrix with pre-trained word vectors from Pyysalo et al., 2013. These word vectors are
obtained from [here](http://evexdb.org/pmresources/vec-space-models/). They were trained using the PubMed abstracts, PubMed Central (PMC), and a Wikipedia dump. 

## Datasets 
We use four biomedical corpora collected by Crichton et al. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). In our implementation, the dataset is accessed via ```../data/```.

## Tagging Scheme
In this study, we use IOBES tagging scheme. `O` denotes non-entity token, `B` denotes the first token of such an entity consisting of multiple tokens, `I` denotes the inside token of the entity, `E` denotes the last token, and `S` denotes a single-token-based entity. We are conducting experiments with IOB tagging scheme at this moment. It will be reposed soon.


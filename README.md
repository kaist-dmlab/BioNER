# DTranNER

## Biomedical Named Entity Recognizer

**DTranNER** is an implementation of a deep-learning-based method suited for biomedical named entity recognition that obtains state-of-the-art performance in NER on the four biomedical benchmark corpora (BC2GM, BC4CHEMD, BC5CDR, and NCBI-Diesease). DTran

## Initial setup

To use **DTranNER**, you need to install Python 2.7, with Numpy, Spacy, and Theano 1.0.0.

## Usage
**To be updated**

## Model Training
```
./train_crf.py --character_model 'cnn' --lstm_hidden_dim 800 --minibatch_size 10 -- test_start 5 --gradient_threshold 20. --dataset 2 --config_model_type 'u_pp' --cnn_case_sensitivity 1 --config_layer_normalization 1 --logging_label_prediction 0 
```

## Download Word Embedding
We initialize the word embedding matrix with pre-trained word vectors from Pyysalo et al., 2013. These word vectors are
obtained from [here](http://evexdb.org/pmresources/vec-space-models/). They were trained using the PubMed abstracts, PubMed Central (PMC), and a Wikipedia dump. 

## Datasets 
We use four biomedical corpora collected by Crichton et al. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). In our implementation, the dataset is accessed through ```../data/```.

## Tagging Scheme
In this study, we use IOBES tagging scheme. `O` denotes non-entity token, `B` denotes the first token of such an entity consisting of multiple tokens, `I` denotes the inside token of the entity, `E` denotes the last token, and `S` denotes a single-token-based entity. 


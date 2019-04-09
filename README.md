# DTranNER

## NER Tagger

DTranNER is an implementation of a deep-learning-based method for biomedical named entity recognition that obtains state-of-the-art performance in NER on the four biomedical benchmark corpora (BC2GM, BC4CHEMD, BC5CDR, and NCBI-Diesease).

## Initial setup

To use DTranNER, you need to install Python 2.7, with Numpy, Spacy, and Theano 1.0.0.


## Model Training

```
./train_crf.py --character_model 'cnn' --lstm_hidden_dim 800 --minibatch_size 10 -- test_start 5 --gradient_threshold 20. --dataset 2 --config_model_type 'u_pp' --cnn_case_sensitivity 1 --config_layer_normalization 1 --logging_label_prediction 0 
```

## Download Word Embedding


## Datasets 
We use four biomedical corpora collected by Crichton et al. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

# DTranNER

## Biomedical Named Entity Recognizer

**DTranNER** is a deep-learning-based method suited for biomedical named entity recognition that obtains state-of-the-art performance in NER on the four biomedical benchmark corpora (BC2GM, BC4CHEMD, BC5CDR, and NCBI-Diesease). **DTranNER** equips with deep learning-based label-label transition model to describe ever-changing contextual relations between neighboring labels.

## Updates
*   **(3 July 2019)** A new version of DTranNER is now available. It is entirely renewed, based on Pytorch, with providing significant performance improvements over the scores on the submitted manuscript.

## Initial setup

To use **DTranNER**, you are required to set up a python3-based environment with packages such as pytorch v1.1.0, numpy, spacy, gensim, and etc. Please refer to the ```requirement.txt```

## Usage
Download the specified word embedding (```wikipedia-pubmed-and-PMC-w2v.bin```) on [here](http://evexdb.org/pmresources/vec-space-models/) and put it under the directory `w2v` whose location is under the project-root directory. 
```
mkdir w2v
mv wikipedia-pubmed-and-PMC-w2v.bin $PROJECT_ROOT/w2v/
```

## Model Training
For model training, we recommend using GPU.
```
python train.py \
    --DTranNER
    --dataset_name ['BC5CDR','BC2GM','BC4CHEMD',or 'NCBI-disease'] \
    --hidden_dim [e.g., 500] \
    --pp_hidden_dim [e.g., 500] \
    --bilinear_dim [e.g., 500] \
    --pp_bilinear_pooling
    --gpu [e.g., 0]
```
You can change the arguments as you want.

## Download Word Embedding
We initialize the word embedding matrix with the pre-trained word vectors from Pyysalo et al., 2013. These word vectors are
obtained from [here](http://evexdb.org/pmresources/vec-space-models/). They were trained using the PubMed abstracts, PubMed Central (PMC), and a Wikipedia dump. 

## Datasets 
We use four biomedical corpora collected by Crichton et al. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). In our implementation, the datasets are accessed via ```$PROJECT_HOME/data/```.

## Tagging Scheme
In this study, we use IOBES tagging scheme. `O` denotes non-entity token, `B` denotes the first token of such an entity consisting of multiple tokens, `I` denotes the inside token of the entity, `E` denotes the last token, and `S` denotes a single-token-based entity. We are conducting experiments with IOB tagging scheme at this moment. It will be reposed soon.

## Contact
Please post a Github issue or contact skhong831@kaist.ac.kr if you have any questions.

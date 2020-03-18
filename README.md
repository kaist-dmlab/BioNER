# DTranNER

## Biomedical Named Entity Recognizer

**DTranNER** is a deep-learning-based method suited for biomedical named entity recognition that obtains state-of-the-art performance in NER on the five biomedical benchmark corpora (BC2GM, BC4CHEMD, BC5CDR-disease, BC5CDR-chemical, and NCBI-Diesease). **DTranNER** equips with deep learning-based label-label transition model to describe ever-changing contextual relations between neighboring labels. Please refer to our paper [DTranNER: biomedical named entity recognition with deep learning-based label-label transition model](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3393-1) for more details.


## Links

- [Initial Setup](#initial-setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)


## Updates
*   **(29 August 2019)** A new version of **DTranNER** is now available. It is entirely renewed, based on PyTorch, with providing significant performance improvements over the scores on the submitted manuscript.


## Initial Setup

To use **DTranNER**, you are required to set up a python3-based environment with packages such as pytorch v1.1.0, numpy, and gensim.


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
    --dataset_name ['BC5CDR-disease','BC5CDR-chem','BC2GM','BC4CHEMD',or 'NCBI-disease'] \
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
Recently, contextualized word embeddings have been emerged. We incorporated **ELMo** https://arxiv.org/abs/1802.05365 into our token embedding layer.


## Datasets 
The source of pre-processed datasets are from https://github.com/cambridgeltl/MTL-Bioinformatics-2016 and
We use biomedical corpora collected by Crichton et al. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). In our implementation, the datasets are accessed via ```$PROJECT_HOME/data/```. For details on NER datasets, please refer to **A Neural Network Multi-Task Learning Approach to Biomedical Named Entity Recognition (Crichton et al. 2017)**.


## Tagging Scheme
In this study, we use IOBES tagging scheme. `O` denotes non-entity token, `B` denotes the first token of such an entity consisting of multiple tokens, `I` denotes the inside token of the entity, `E` denotes the last token, and `S` denotes a single-token-based entity. We are conducting experiments with IOB tagging scheme at this moment. It will be reported soon.


## Benchmarks

Here we compare our model with recent state-of-the-art models on the five biomedical corpora mentioned above. We measure F1 score as the evaluation metric. The experimental results are shown in below the table.

|Model | [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES) | [BC4CHEMD](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC4CHEMD-IOBES) | [BC5CDR-Chemical](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-chem-IOBES) | [BC5CDR-Disease](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-disease-IOBES) | [NCBI-disease](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/NCBI-disease-IOBES)|
| ---- | ---- | ---- | ---- | ---- | ---- |
| [Att-BiLSTM-CRF 2017](https://github.com/lingluodlut/Att-ChemdNER) | - | 91.14 | 92.57 | - | - |
| [D3NER 2018](https://github.com/trangnm58/D3NER) | - | - | 93.14 | 84.68 | 84.41 |
| [Collabonet 2018](https://github.com/wonjininfo/CollaboNet) | 79.73 | 88.85 | 93.31 | 84.08 | 86.36 |
| [Wang et al. 2018](https://github.com/yuzhimanhua/Multi-BioNER) | 80.74 | 89.37 | 93.03 | 84.95 | 86.14 |
| [BioBERT v1.0](https://github.com/dmis-lab/biobert) | 84.40 | 91.41 | 93.44 | 86.56 | 89.36 |
| [BioBERT v1.1](https://github.com/dmis-lab/biobert) | **84.72** | **92.36** | 93.47 | 87.15 | **89.71** |
| DTranNER | 84.56 | 91.99 | **94.16** | **87.22** | 88.62 |


## Contact
Please post a Github issue or contact skhong831@kaist.ac.kr or skhong0831@gmail.com if you have any questions.

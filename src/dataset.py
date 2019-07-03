import os
import sys
import csv
import math
import time
import pickle
import numpy as np
import gensim
import nltk
import spacy
from torch.utils.data import Dataset

class PreprocessingPOS:
    __TAGS__ = ['NOUN', 'VERB', 'ADJ', 'ADP', 'DET', 'PROPN', 'PUNCT', 'SYM', 'CCONJ', 'NUM', 'X', 'PART', 'INTJ', 'SPACE', 'PRON', 'AUX', 'ADV', 'SPACE', 'FAULT']
    n_fault_pos_tagging_sentences = 0
    def __init__(self):
#         self.spacy_nlp = spacy.load('en_core_web_sm')
        self.spacy_nlp = spacy.load('en_core_sci_md')
        self.pos_dict = dict()
        self.pos_dict_reversed = dict()
        for idx, tag in enumerate(PreprocessingPOS.__TAGS__):
            self.pos_dict[tag] = idx
            self.pos_dict_reversed[idx] = tag
    
    @classmethod
    def get_pos_tags(cls):
        return cls.__TAGS__
    
    def __len__(self):
        return len(PreprocessingPOS.__TAGS__)
    
    def __call__(self, sentence):
        pre_sentence = ' '.join(sentence)
#         print(sentence, pre_sentence)
        doc = self.spacy_nlp(pre_sentence)
        pos_sentence = list()
        for token in doc:
            pos_sentence.append(self.pos_dict[token.pos_])
        if len(sentence) != len(pos_sentence):
            PreprocessingPOS.n_fault_pos_tagging_sentences += 1
#             print(sentence)
            print("POS Error!!")
            pos_sentence = [self.pos_dict['FAULT']] * len(sentence)
        return pos_sentence

    def annotate_pos(self, sentence):
        pre_sentence = ' '.join(sentence)
        doc = self.spacy_nlp(pre_sentence)
        pos_sentence = list()
        for token in doc:
            pos_sentence.append(token.pos_)
        return pos_sentence
        
    
class DatasetPreprosessed(Dataset):
    __START_TAG__ = "<START>"
    __STOP_TAG__ = "<STOP>"
    __MODEL_FILEPATH__ = "../w2v/wikipedia-pubmed-and-PMC-w2v.bin"
    __CHARACTER_VOCABULARY__ = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
#     __CHARACTER_VOCABULARY__ = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    __DATA_PATH__ = "../data/"
    __DATASETS__ = {"bioes":['BC2GM-IOBES/', 'BC4CHEMD-IOBES/', 'BC5CDR-IOBES/', 'BC5CDR-chem-IOBES/', 'BC5CDR-disease-IOBES/', 'JNLPBA-IOBES/', 'NCBI-disease-IOBES/', 'Ex-PTM-IOBES/', 'linnaeus-IOBES/'],
                "bio":['BC2GM-IOB/', 'BC4CHEMD/', 'BC5CDR-IOB/', 'BC5CDR-chem-IOB/', 'BC5CDR-disease-IOB/', 'JNLPBA/', 'NCBI-disease-IOB/', 'Ex-PTM-IOB/', 'linnaeus-IOB/']
                }
    __FILE_LIST__ = ['train.tsv', 'devel.tsv', 'test.tsv']

    def __init__(self, tagging_scheme, dataset_name, model_type, embedding_dim, character_token_valid_max_length, train=True):
        self.model_type = model_type
        self.train = train
        self.embedding_dim = embedding_dim
        samples = list()
        self.sentences = list()
        self.labels = list()
        self.label_dict = dict()
        self.label_dict_reversed = dict()
        self._unknown = np.zeros((self.embedding_dim,), dtype=np.float32) # np.random.uniform(-0.00001, 0.00001, (self.embedding_dim,))
        self._start = np.full((self.embedding_dim,), 1.)
        self._end = np.full((self.embedding_dim,), -1.)
        self.max_length = character_token_valid_max_length
        self.pos_processor = PreprocessingPOS()
        
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(DatasetPreprosessed.__MODEL_FILEPATH__, binary=True)
        self.word2vec = w2v_model.wv
        self.character_matrix = np.identity(len(DatasetPreprosessed.__CHARACTER_VOCABULARY__), dtype=np.float32)
        print("# %d of characters" %len(DatasetPreprosessed.__CHARACTER_VOCABULARY__))
        
        for dataset in DatasetPreprosessed.__DATASETS__[tagging_scheme]:
            if dataset.startswith(dataset_name):
                self.dataset = dataset
                break
            
        _file = DatasetPreprosessed.__FILE_LIST__[:-1] if train else DatasetPreprosessed.__FILE_LIST__[-1] 
        if isinstance(_file, list):
            for file_name in _file:
                samples.extend(self.read_sentence_from_file(DatasetPreprosessed.__DATA_PATH__ + dataset + file_name))
        else:
            samples = self.read_sentence_from_file(DatasetPreprosessed.__DATA_PATH__ + dataset + _file)
        
        self.length = len(samples)
        print('# of sentences in {} = {}'.format(dataset, self.length))
            
        for sample in samples:
            _sentence = []
            _label = []
            for i, token in enumerate(sample):
                if i%2 == 0:
                    _sentence.append(token)
                else:
                    _label.append(token)
            if len(_sentence) < 2:
                _sentence.append('.')
                _label.append('O')
            assert(len(_sentence) == len(_label))
            self.sentences.append(_sentence)
            self.labels.append(_label)
        
        self.build_label_dict()
    
    @classmethod
    def character_vocabulary_size(cls):
        return len(cls.__CHARACTER_VOCABULARY__)
    
    def build_label_dict(self):
        label_set = set()
        for labels in self.labels:
            label_set.update(labels)
        for i, label in enumerate(list(label_set)):
            self.label_dict[label] = i
            self.label_dict_reversed[i] = label

    def read_sentence_from_file(self, path):
        assert isinstance(path, str)
        sentences = list()
        sentence = list()
        with open(path) as fin:
            reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                if i == 0:
                    pass
                if len(row) < 2 and len(sentence) != 0:
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(row[0])
                    sentence.append(row[1])
        
        return sentences
                
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        sentence_vectors = list()
        character_level_sentence_vectors = list()
        for w in sentence:
            try:
#                 vec = self.word2vec[w.lower()]
                vec = self.word2vec[w]
                sentence_vectors.append(vec)
            except KeyError:
#                 print("W2V=%s in Key Error" %w)
#                 print(sentence)
#                 try:
#                     vec = self.word2vec[w.lower()]
#                 except KeyError:
#                     print("W2V=%s in Key Error 2" %w)
                sentence_vectors.append(self._unknown)
#                 try:
# #                     print("W2V=%s" %w)
#                     sentence_vectors.append(vec)
#                 except KeyError:
# #                     print("W2V=%s in Key Error" %w)
#                     sentence_vectors.append(self._unknown)
            token_character_vectors = list()
            characters = np.array([self.character_matrix[DatasetPreprosessed.__CHARACTER_VOCABULARY__.index(i)] for i in w if i in DatasetPreprosessed.__CHARACTER_VOCABULARY__], dtype=np.float32)

            if len(characters) > self.max_length:
                characters = characters[:self.max_length]
            elif 0 < len(characters) < self.max_length:
                characters = np.concatenate((characters, np.zeros((self.max_length - len(characters), len(DatasetPreprosessed.__CHARACTER_VOCABULARY__)), dtype=np.float32)))
            elif len(characters) == 0:
                characters = np.zeros((self.max_length, len(DatasetPreprosessed.__CHARACTER_VOCABULARY__)), dtype=np.float32)
            character_level_sentence_vectors.append(characters)

        pos_sentence = self.pos_processor(sentence)
        
        data = np.array(sentence_vectors, dtype=np.float32)
        label = np.array([self.label_dict[l] for l in self.labels[index]], dtype=np.int32)
        character_data = np.array(character_level_sentence_vectors, dtype=np.float32)
        pos = np.array(pos_sentence, dtype=np.int64)
#         print(len(pos_sentence), pos.shape, data.shape, character_data.shape)

        if self.model_type:
            pp_sentence_vectors = sentence_vectors[:]
            pp_sentence_vectors.insert(0, self._start)
            pp_sentence_vectors.append(self._end)
            pp_pos_sentence = pos_sentence[:]
            pp_pos_sentence.insert(0, self.pos_processor.pos_dict['X'])
            pp_pos_sentence.append(self.pos_processor.pos_dict['X']) 
            assert len(sentence_vectors) != len(pp_sentence_vectors)
            pp_data = np.array(pp_sentence_vectors, dtype=np.float32)
            pp_pos = np.array(pp_pos_sentence, dtype=np.int64)
            return (sentence, data, pp_data, pos, pp_pos, character_data, label) if not self.train else (data, pp_data, pos, pp_pos, character_data, label)
#             return (sentence, data, pp_data, character_data, label) if not self.train else (data, pp_data, character_data, label)
        
        return (sentence, data, pos, character_data, label) if not self.train else (data, pos, character_data, label) 

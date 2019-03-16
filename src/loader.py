import os
import sys
import argparse
import csv
import pickle
import numpy as np
import spacy

class Loader:
    def __init__(self):
        pass
    
    def write_all_NERs(self, data_path, dataset, train_file):
        file_name = '_NERs.txt'
        fout = open(dataset[:-1] + train_file[:-3] + file_name, 'w', 0)
        entity = []
        sentence = []
        with open(data_path + dataset + train_file) as fin:
            reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
#                 if i is 0 or len(row) is 0:
                if len(row) is 0:
                    fout.write(' '.join(sentence) + '\n\n')
                    sentence = []
                    continue
                if row[1].startswith('S'):
                    fout.write(row[0] + ' ')
                    fout.write(row[1] + '\n')
                elif row[1].startswith('B') or row[1].startswith('I'):
                    entity.append(row[0])
                    entity.append(row[1])
                    name = ' '.join(entity)
                    fout.write(name + '\n')
                    entity = []
                elif row[1].startswith('E'):
                    entity.append(row[0])
                    entity.append(row[1])
                    name = ' '.join(entity)
                    fout.write(name + '\n')
                    entity = []
                sentence.append(row[0])
        
        fout.close()
        
        return

    def gathering_all_words(self, data_path, dataset, train_file):
        train_sentences = []
        train_labels = []
        sentence = []
        label = []
        all_labels = []
        with open(data_path + dataset + train_file) as fin:
            reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
#                 if i == 0:
#                     print row[0]
#                     print row[1]
#                     continue
                if len(row) == 0:
                    assert(len(sentence) == len(label))
    #                 print sentence
    #                 print label
                    train_sentences.append(sentence)
                    train_labels.append(label)
                    sentence = []
                    label = []
                else:
    #                 sentence.append(row[0].lower())
                    sentence.append(row[0])
                    label.append(row[1])
    #                 all_labels.append(row[1])
        
        print('# of sentences in {} = {}'.format(dataset, len(train_sentences)))
        print('# of labels in {} = {}'.format(dataset, len(train_labels)))
    #     print('All labels = ', set(all_labels))
        
        return train_sentences, train_labels

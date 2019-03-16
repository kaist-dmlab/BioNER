import os
import sys
import csv
import pickle
import collections
import numpy as np

class CharacterProcessing:
    def __init__(self):
        self.characters = set()
        self.n_characters = 0
        self.max_token_length = 0
        self.char2idx = collections.OrderedDict()
        self.idx2char = collections.OrderedDict()
#         self.start_char = '<START_CHAR>'
        self.start_char = '<START_CHAR>'
        self.end_char = '<END_CHAR>' 
        self.max_sentence_length = 0
            
    def build_train_characters(self, train_sentences, max_sentence_length):
        self.max_sentence_length = max_sentence_length
#         self.characters.add(self.start_char)
        self.characters.add(self.start_char)
        self.characters.add(self.end_char)
        max_length = 0
        for each_file in train_sentences:
            for sentence in each_file:
                sentence_characters = []
                for token in sentence:
#                     if token == '-':
#                         continue
                    length = 0
                    if token.isdigit():
                        token = '0'
                    for c in token.lower():
                        length += 1
                        if c.isdigit():
                            c = '0'
                        self.characters.add(c)
                    if max_length < length:
                        max_length = length
        
        self.max_token_length = max_length + 2
        self.n_characters = len(self.characters)
        
        for i, char in enumerate(list(self.characters)):
            self.char2idx[char] = i
            self.idx2char[i] = char
            
        assert(len(self.char2idx) == len(self.idx2char) == len(self.characters))

        # Index of END_CHAR is 0
        print 'start char index = ', self.char2idx[self.start_char]
        print 'end char index = ', self.char2idx[self.end_char]
        print 'max token length = ', self.max_token_length
#         for i in xrange(len(self.idx2char)):
#             print 'end char = ', self.idx2char[i]
        
        return self.n_characters, self.char2idx[self.start_char], self.char2idx[self.end_char]

    def build_train_characters_LSTM(self, train_sentences, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        max_length = 0
        for each_file in train_sentences:
            for sentence in each_file:
                sentence_characters = []
                for token in sentence:
                    length = 0
                    if token.isdigit():
                        token = '0'
                    for c in token.lower():
                        length += 1
                        if c.isdigit():
                            c = '0'
                        self.characters.add(c)
                    if max_length < length:
                        max_length = length
        
        self.max_token_length = max_length
        self.n_characters = len(self.characters)
        
        for i, char in enumerate(list(self.characters)):
            self.char2idx[char] = i
            self.idx2char[i] = char
            
        assert(len(self.char2idx) == len(self.idx2char) == len(self.characters))
        print 'max token length = ', self.max_token_length
        
        return self.n_characters, 0, 0

    def build_train_characters_with_case_sensitivity_LSTM(self, train_sentences, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        max_length = 0
        for each_file in train_sentences:
            for sentence in each_file:
                sentence_characters = []
                for token in sentence:
                    length = 0
                    if token.isdigit():
                        token = '0'
                    for c in token:
                        length += 1
                        if c.isdigit():
                            c = '0'
                        self.characters.add(c)
                    if max_length < length:
                        max_length = length
        
        self.max_token_length = max_length
        self.n_characters = len(self.characters)
        
        for i, char in enumerate(list(self.characters)):
            self.char2idx[char] = i
            self.idx2char[i] = char
            
        assert(len(self.char2idx) == len(self.idx2char) == len(self.characters))
        print 'max token length = ', self.max_token_length
        
        return self.n_characters, 0, 0

    def build_train_characters_with_case_sensitivity_for_CNN(self, train_sentences, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        self.characters.add(self.start_char)
        self.characters.add(self.end_char)
        max_length = 0
        for each_file in train_sentences:
            for sentence in each_file:
                for token in sentence:
                    length = 0
                    if token.isdigit():
                        token = '0'
                    for c in token:
                        if c.isdigit():
                            c = '0'
                        length += 1
                        self.characters.add(c)
                    if max_length < length:
                        max_length = length
        
        self.max_token_length = max_length + 2
        self.n_characters = len(self.characters)
        
        for i, char in enumerate(list(self.characters)):
            self.char2idx[char] = i
            self.idx2char[i] = char
            
        assert(len(self.char2idx) == len(self.idx2char) == len(self.characters))

        print 'start char index = ', self.char2idx[self.start_char]
        print 'end char index = ', self.char2idx[self.end_char]
        print 'max token length = ', self.max_token_length
        
        return self.n_characters, self.char2idx[self.start_char], self.char2idx[self.end_char]

    def build_train_characters_dataset(self, train_sentences):
        sentences = []
        lengths = []
        upper_sentences = []
        position_sentences = []   # start_character=1, end_character=2, else_character=0
        for sentence in train_sentences:
            one_sentence = [[0] * self.max_token_length for i in xrange(self.max_sentence_length)]
            length = [0] * self.max_sentence_length
            for i, token in enumerate(sentence):
                token_characters = [self.char2idx[self.end_char]] * self.max_token_length
                if token.isdigit():
                    token = '0'
                for j, char in enumerate(token.lower()):
                    if char.isdigit():
                        char = '0'
                    token_characters[j+1] = self.char2idx[char]
                token_characters[0] = self.char2idx[self.start_char]
                length[i] = len(token) + 2
                one_sentence[i] = token_characters
            sentences.append(one_sentence)
            lengths.append(length)
        
        return sentences, lengths
    
    def build_train_characters_dataset_with_case_sensitivity_for_LSTM(self, train_sentences):
        sentences = []
        lengths = []
        for sentence in train_sentences:
            one_sentence = [[0] * self.max_token_length for i in xrange(self.max_sentence_length)]
            length = [0] * self.max_sentence_length
            for i, token in enumerate(sentence):
                token_characters = [0] * self.max_token_length
                if token.isdigit():
                    token = '0'
                for j, char in enumerate(token):
                    if char.isdigit():
                        char = '0'
                    token_characters[j] = self.char2idx[char]
                length[i] = len(token)   # concatenated start and end characters
                one_sentence[i] = token_characters
            sentences.append(one_sentence)
            lengths.append(length)
        
        return sentences, lengths

    def build_train_characters_dataset_with_case_sensitivity_for_CNN(self, train_sentences):
        sentences = []
        lengths = []
        for sentence in train_sentences:
            one_sentence = [[self.char2idx[self.end_char]] * self.max_token_length for i in xrange(self.max_sentence_length)]
            length = [0] * self.max_sentence_length
            for i, token in enumerate(sentence):
                token_characters = [self.char2idx[self.end_char]] * self.max_token_length
                if token.isdigit():
                    token = '0'
                for j, char in enumerate(token):
                    if char.isdigit():
                        char = '0'
                    token_characters[j+1] = self.char2idx[char]
                token_characters[0] = self.char2idx[self.start_char] 
                length[i] = len(token) + 2   # concatenated start and end characters
                one_sentence[i] = token_characters
            sentences.append(one_sentence)
            lengths.append(length)
        
        return sentences, lengths
            
    def build_train_characters_dataset_LSTM(self, train_sentences):
        sentences = []
        lengths = []
        upper_sentences = []
        position_sentences = []   # start_character=1, end_character=2, else_character=0
        for sentence in train_sentences:
            one_sentence = [[0] * self.max_token_length for i in xrange(self.max_sentence_length)]
            length = [0] * self.max_sentence_length
            for i, token in enumerate(sentence):
                token_characters = [0] * self.max_token_length
                if token.isdigit():
                    token = '0'
                for j, char in enumerate(token.lower()):
                    if char.isdigit():
                        char = '0'
                    token_characters[j] = self.char2idx[char]
                length[i] = len(token)
                one_sentence[i] = token_characters
            sentences.append(one_sentence)
            lengths.append(length)
        
        return sentences, lengths

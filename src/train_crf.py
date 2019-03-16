import os
import sys
import argparse
import csv
import math
import time
# import BiLSTMCNN as charcnn
# import DebugBiLSTMCNN as charcnn
# import Debug2LayerBiLSTMCNNW2VHinge as charcnn
# import DoubleLayerBiLSTMCNNW2VHighway as charcnn
# import CNNDoubleBiLSTMCNNW2V as charcnn
# import CRFDoubleBILSTMW2VCNN as charcnn
# import CRFPPBiLSTMW2VCNN as crfpp
# import AttCRFPPBiLSTMW2VCNN as attcrfpp
import InitializeModel as initialize
import pickle
import character
import numpy as np
import loader
import Evaluation as conlleval
import spacy
import loadw2v 

class DatasetPreprocessing:
    def __init__(self, path, dataset, train_file, n_sampling=2, training=True):
        self.dataset = dataset[:-1]
        test_sentences = []
        sentences = []
        labels = []
        if isinstance(train_file, list):
            for file in train_file:
                test_sentences.extend(read_sentence_from_file(path + dataset + file))
        elif isinstance(train_file, str):
            test_sentences = read_sentence_from_file(path + dataset + train_file)
        
        print('# of sentences in {} = {}'.format(dataset, len(test_sentences)))
            
        self.sentence_label = []
        n_oversampled = 0
        for sentence in test_sentences:
            _sentence = []
            _label = []
            for i, token in enumerate(sentence):
                if i%2 == 0:
                    _sentence.append(token)
                else:
                    _label.append(token)
            if len(_sentence) < 2:
                _sentence.append('.')
                _label.append('O')
            assert(len(_sentence) == len(_label))
            if training == True and self.dataset != 'BC2GM-IOBES':
                score = self.scoring(_label)
#                 if score >= 0.8:
                if score >= 100:
                    n_oversampled += 1
                    for i in range(n_sampling):
                        self.sentence_label.append((_sentence, _label))
            self.sentence_label.append((_sentence, _label))

        print "N Oversampled = ", n_oversampled
        
    def scoring(self, labels):
        eps = 1e-5
        score = 0.
        n_O = 0
        n_non_O = 0
        if self.dataset == 'BC5CDR-IOBES':
            for label in labels:
                if label.startswith('B-C') or label.startswith('E-C'):
                    score += 0.25
                if label.startswith('B-D') or label.startswith('I-D') or label.startswith('E-D'):
                    score += 0.15
                if label.startswith('I-C'):
                    score += 0.1
                if label.startswith('S'):
                    score -= 5.
                if label.startswith('O'):
                    n_O += 1
                else:
                    n_non_O += 1
        
        score += n_non_O / (n_O + eps)
        score = score if len(labels) >= 5 else 0

        return score
        
    def get_dataset_sentences(self):
#         sentences_permutated = np.random.permutation(self.sentence_label)
#         sentences_permutated = self.sentence_label
        sentences = []
        labels = []
        max_sentence_length = 0
        for sentence, label in self.sentence_label:
            sentences.append(sentence)
            labels.append(label)
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence) 
        assert len(sentences) == len(labels)
        print('# of training sentences and labels and MAX sentence length = {} and {} and {}'.format(len(sentences), len(labels), max_sentence_length))            
        return sentences, labels
    
def create_vocab(files):
    words = []
    for each_file in files:
        for sentence in each_file:
            words.extend(sentence)
    vocab = set(words)
    
    return vocab            

def read_sentence_from_file(path):
    assert isinstance(path, str)
    test_sentences = []
    test_sentence = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == 0:
                pass
            if len(row) < 2:
                test_sentences.append(test_sentence)
                test_sentence = []
            else:
                _token = row[0]
                _label = row[1]
                if _token.isdigit():
                    _token = '0'
                test_sentence.append(_token)
                test_sentence.append(_label)
    
    return test_sentences


def create_binary_training_samples(_sentences, _labels, word2idx, MAX_SENTENCE_LEN, MAX_WORD_IDX):
    train_sentences = []
    train_sentences_lengths = []
    train_labels = []
    sentence = []
    label = []
    counter = [0] * 5
    
    assert(len(_sentences) == len(_labels))
    for i, _sentence in enumerate(_sentences):
        sentence = [0] * MAX_SENTENCE_LEN
        label = [0] * MAX_SENTENCE_LEN
        for index, row in enumerate(zip(_sentence, _labels[i])):
            _token = row[0]
            if _token.isdigit():
                _token = '0'
            try:
                token_idx = word2idx[_token.lower()]
            except KeyError:
                token_idx = 0
            sentence[index] = token_idx
            _label = row[1]
            if _label.startswith('B'):
                value = 1
            elif _label.startswith('I'):
                value = 2
            elif _label.startswith('E'):
                value = 3
            elif _label.startswith('S'):
                value = 4
            else:
                value = 0
            counter[value] += 1    
            label[index] = value

        sentence[index+1] = MAX_WORD_IDX
        train_sentences.append(sentence)
        train_sentences_lengths.append(index)
        train_labels.append(label)

    counter_str = ' '.join([str(entry) for entry in counter])
    print('counter of label class = {}'.format(counter_str))
    
    for sentence in train_sentences:
        count = 0
        for token in sentence:
            if token >= 67000:
                count += 1
        if count != 1:
            print('Found Problem')
            break
                
    return train_sentences, train_sentences_lengths, train_labels

def create_testing_sentences(path, dataset, train_file):
    test_sentences = read_sentence_from_file(path + dataset + train_file)
    print('# of sentences in {} = {}'.format(dataset, len(test_sentences)))
    max_sentence_length = 0    
    sentences = []
    labels = []
    for sentence in test_sentences:
        _sentence = []
        _label = []
        for i, token in enumerate(sentence):
            if i%2 == 0:
                _sentence.append(token)
            else:
                _label.append(token)
        if len(_sentence) < 2:
            _sentence.append('.')
            _label.append('O')
        assert(len(_sentence) == len(_label))
        if len(_sentence) > max_sentence_length:
            max_sentence_length = len(_sentence) 
        sentences.append(_sentence)
        labels.append(_label)
        
    assert len(sentences) == len(labels)
    print('# of testing sentences and labels and MAX sentence length = {} and {} and {}'.format(len(sentences), len(labels), max_sentence_length))            
    return sentences, labels

def create_training_samples(_dataset, _sentences, _labels, word2idx, nlp, pos2idx, MAX_SENTENCE_LEN, MAX_WORD_IDX):
    train_sentences = []
    train_sentences_lengths = []
    pos_sentences = []
    train_labels = []
    train_pos = []
    sentence = []
    label = []
    
    if _dataset == 0 or _dataset == 1 or _dataset == 3 or _dataset == 4 or _dataset == 6 or _dataset == 7 or _dataset == 8:
        counter = [0] * 5
    elif _dataset == 2:
        counter = [0] * 9
    elif _dataset == 5:
        counter = [0] * 21
     
    assert(len(_sentences) == len(_labels))
     
    for i, _sentence in enumerate(_sentences):
        key_error = False
        sentence = [0] * MAX_SENTENCE_LEN
        label = [0] * MAX_SENTENCE_LEN
        pos_sentence = [0] * MAX_SENTENCE_LEN
        __sentence = ' '.join(_sentence)
        doc = nlp(__sentence.decode('utf-8'))
        for j, token in enumerate(doc):
#             pos_sentence[j] = pos2idx[token.pos_.encode('ascii', 'ignore')]
            try:
                pos_sentence[j] = pos2idx[token.tag_.encode('ascii', 'ignore')]
            except IndexError:
                print 'Exception Occurred'
            except KeyError:
                print 'Key Error Exception Occurred', token, '  ', token.tag_.encode('ascii', 'ignore')
                key_error = True
        
        if key_error == True:
            print 'Bypass due to key error'
            continue
        
        for index, row in enumerate(zip(_sentence, _labels[i])):
            _token = row[0]
            if _token.isdigit():
                _token = '0'
            try:
                token_idx = word2idx[_token.lower()]
            except KeyError:
                token_idx = 0
            sentence[index] = token_idx
            _label = row[1]
            if _dataset == 2:
                if _label.startswith('B-D'):
                    value = 1
                elif _label.startswith('I-D'):
                    value = 2
                elif _label.startswith('E-D'):
                    value = 3
                elif _label.startswith('S-D'):
                    value = 4
                elif _label.startswith('B-C'):
                    value = 5
                elif _label.startswith('I-C'):
                    value = 6
                elif _label.startswith('E-C'):
                    value = 7
                elif _label.startswith('S-C'):
                    value = 8
                else:
                    value = 0
            elif _dataset == 5:
                if _label.startswith('B-protein'):
                    value = 1
                elif _label.startswith('I-protein'):
                    value = 2
                elif _label.startswith('E-protein'):
                    value = 3
                elif _label.startswith('S-protein'):
                    value = 4
                elif _label.startswith('B-cell_type'):
                    value = 5
                elif _label.startswith('I-cell_type'):
                    value = 6
                elif _label.startswith('E-cell_type'):
                    value = 7
                elif _label.startswith('S-cell_type'):
                    value = 8
                elif _label.startswith('B-cell_line'):
                    value = 9
                elif _label.startswith('I-cell_line'):
                    value = 10
                elif _label.startswith('E-cell_line'):
                    value = 11
                elif _label.startswith('S-cell_line'):
                    value = 12
                elif _label.startswith('B-DNA'):
                    value = 13
                elif _label.startswith('I-DNA'):
                    value = 14
                elif _label.startswith('E-DNA'):
                    value = 15
                elif _label.startswith('S-DNA'):
                    value = 16
                elif _label.startswith('B-RNA'):
                    value = 17
                elif _label.startswith('I-RNA'):
                    value = 18
                elif _label.startswith('E-RNA'):
                    value = 19
                elif _label.startswith('S-RNA'):
                    value = 20
                else:
                    value = 0
            elif _dataset == 1:
                if _label.startswith('B-C'):
                    value = 1
                elif _label.startswith('I-C'):
                    value = 2
                elif _label.startswith('E-C'):
                    value = 3
                elif _label.startswith('S-C'):
                    value = 4
                else:
                    value = 0
            elif _dataset == 0:
                if _label.startswith('B-G'):
                    value = 1
                elif _label.startswith('I-G'):
                    value = 2
                elif _label.startswith('E-G'):
                    value = 3
                elif _label.startswith('S-G'):
                    value = 4
                else:
                    value = 0
            elif _dataset == 7:
                if _label.startswith('B-P'):
                    value = 1
                elif _label.startswith('I-P'):
                    value = 2
                elif _label.startswith('E-P'):
                    value = 3
                elif _label.startswith('S-P'):
                    value = 4
                else:
                    value = 0
            elif _dataset == 3:
                if _label.startswith('B-C'):
                    value = 1
                elif _label.startswith('I-C'):
                    value = 2
                elif _label.startswith('E-C'):
                    value = 3
                elif _label.startswith('S-C'):
                    value = 4
                else:
                    value = 0
            elif _dataset == 4 or _dataset == 6:
                if _label.startswith('B-D'):
                    value = 1
                elif _label.startswith('I-D'):
                    value = 2
                elif _label.startswith('E-D'):
                    value = 3
                elif _label.startswith('S-D'):
                    value = 4
                else:
                    value = 0
            elif _dataset == 8:
                if _label.startswith('B-S'):
                    value = 1
                elif _label.startswith('I-S'):
                    value = 2
                elif _label.startswith('E-S'):
                    value = 3
                elif _label.startswith('S-S'):
                    value = 4
                else:
                    value = 0

            counter[value] += 1    
            label[index] = value
 
#         sentence[index+1] = MAX_WORD_IDX
        if len(sentence) == 0:
            print '!!! Zero length sentence !!!'
        train_sentences.append(sentence)
#         print sentence[index], index, word2idx['.']
        train_sentences_lengths.append(index+1)
        train_labels.append(label)
        train_pos.append(pos_sentence)
 
    counter_str = ' '.join([str(entry) for entry in counter])
    print('counter of label class = {}'.format(counter_str))
     
    for sentence in train_sentences:
        count = 0
        for token in sentence:
            if token >= MAX_WORD_IDX:
                print token
                count += 1
        if count > 0:
            print('!!!Found Problem!!!')
            break
      
    return train_sentences, train_sentences_lengths, train_labels, train_pos

def random_selection(pre_training_sentences, pre_training_labels, proportion):
    n = len(pre_training_sentences)
    assert(len(pre_training_sentences) == len(pre_training_labels))
    
    c = np.int(n * proportion)
    indexes = np.random.choice(n, c)
    indexes = indexes.tolist()
    sampled_train_set = []
    sampled_label_set = []
    for i in indexes:
        sampled_train_set.append(pre_training_sentences[i])
        sampled_label_set.append(pre_training_labels[i])
    
    return sampled_train_set, sampled_label_set

def random_permutation(training_sentences, training_labels, sampled_train_set, sampled_label_set, curr_task_index, prev_task_index):
    n = len(training_sentences)
    assert(len(training_sentences) == len(training_labels))
    
    c = len(sampled_train_set)
    assert(len(sampled_train_set) == len(sampled_label_set))
    
    total_n_samples = n + c
    task_indexes = [curr_task_index] * (total_n_samples)
    
    indexes = np.random.choice(n, c)
    for i, index in enumerate(indexes):
        sample = training_sentences[index]
        label = training_labels[index]
        training_sentences[index] = sampled_train_set[i]
        training_labels[index] = sampled_label_set[i]
        training_sentences.append(sample)
        training_labels.append(label)
        task_indexes[index] = prev_task_index
    
    assert(len(training_sentences) == len(training_labels) == total_n_samples)
    
    return training_sentences, training_labels, task_indexes

def random_shuffle_train_dataset(training_dataset, dataset_idx, word2idx, nlp, pos2idx, read_characters, case_sensitivity, config_lstm_character, MAX_SENTENCE_LEN, MAX_WORD_IDX):
    _sentences, _labels = training_dataset.get_dataset_sentences()
    sentences, lengths, labels, pos = create_training_samples(dataset_idx, _sentences, _labels, word2idx, nlp, pos2idx, MAX_SENTENCE_LEN, MAX_WORD_IDX)
    if case_sensitivity == True:
        if config_lstm_character == True:
            train_character_dataset, train_char_lengths = read_characters.build_train_characters_dataset_with_case_sensitivity_for_LSTM(_sentences)
        else:
            train_character_dataset, train_char_lengths = read_characters.build_train_characters_dataset_with_case_sensitivity_for_CNN(_sentences)
    else:
        if config_lstm_character == True:
            train_character_dataset, train_char_lengths = read_characters.build_train_characters_dataset_LSTM(_sentences)
        else:
            train_character_dataset, train_char_lengths = read_characters.build_train_characters_dataset(_sentences)
    return sentences, lengths, labels, train_character_dataset, train_char_lengths, pos

def print_configuration(args):
    print 'input_mlp_dim = ', args.input_mlp_dim
    print 'lstm_hidden_dim = ', args.lstm_hidden_dim
    print 'lstm2 hidden dim = ', args.lstm2_hidden_dim
    print 'lstm_pp_hidden_dim = ', args.lstm_pp_hidden_dim
    print 'lstm_pp_type_hidden_dim = ', args.lstm_pp_type_hidden_dim
    print 'lstm_pp_decoder_hidden_dim = ', args.lstm_pp_decoder_hidden_dim
    print 'lstm_pp_type_decoder_hidden_dim = ', args.lstm_pp_type_decoder_hidden_dim
    print 'mlp_dim = ', args.mlp_dim
    print 'pp_mlp_dim = ', args.pp_mlp_dim
    print 'pp_mlp_dim2 = ', args.pp_mlp_dim2
    print 'pp_type_mlp_dim = ', args.pp_type_mlp_dim
    print 'dropout_rate_x = ', args.dropout_rate_x
    print 'dropout_rate_s = ', args.dropout_rate_s
    print 'minibatch size = ', args.minibatch_size
    print 'test starts = ', args.test_start
    print 'method = ', args.method
    print 'dataset = ', args.dataset
    print 'character model = ', args.character_model
    if args.character_model == 'lstm':
        print 'lstm char hidden dim = ', args.lstm_char_hidden_dim
    
    if args.pairwise_potential > 0:
        print 'pairwise potential turned On'
        if args.separate_pairwise_potential > 0:
            print 'separate pairwise potential turned On'
        if args.pairwise_potential_encoder_decoder > 0:
            print 'pairwise_potential_encoder_decoder turned On'
        if args.w2v_pairwise_potential > 0:
            print 'pairwise potential has w2v'
        if args.sentence_level_phrase_transition > 0:
            print 'Sentence Level Phrase Pairwise Transition On'
        if args.sentence_level_type_transition > 0:
            print 'Sentence Level Type Pairwise Transition On' 
        if args.pp_type_sentence_embedding_status == 2:
            print 'Combined BiLSTM Max Pooling On'
        elif args.pp_type_sentence_embedding_status == 1:
            print 'BiLSTM Max Pooling On' 
    if args.forward_backward > 0:
        print 'Forward Backward Training On'
    if args.bilstm_2_added > 0:
        print 'BiLSTM 2 added'
    if args.dropout_on > 0:
        print 'Dropout On'
    if args.orthogonal_weight_init > 0:
        print 'Orthogonal_weight_init On'
    if args.weight_decay > 0:
        print 'L2 weight decay On with ', args.weight_decay_rate
    if args.logging_label_prediction > 0:
        print 'Logging label predictions'
    if args.exclusive_begin_end_tagging > 0:
        print 'Excluding Begin End Tagging'
    if args.optimization == 'adam':
        print 'Adam optimization'
    else:
        print 'Momentum optimization'
    if args.gradient_threshold > 0.:
        print 'Gradient Threshold = ', args.gradient_threshold
    if args.final_mlp_added > 0:
        print 'MLP final layer added on LSTM'
    if args.shared_cnn > 0:
        print 'A CNN is shared'
    if args.config_gru > 0:
        print 'GRU On'
    if args.input_mlp > 0:
        print 'Input FeedForward Layer On with ', args.input_mlp_activiation.upper()
    if args.config_initial_hidden_modeling > 0:
        print 'Init LSTM state modeling On'
    if args.pairwise_output > 0:
        print 'Pairwise Output Modeling On'
    if args.config_pp_joint_training > 0:
        print 'PP Joint Training On'
    if args.config_crf_pp_training > 0:
        print 'PP Only Training On'
    if args.config_no_crf_training > 0:
        print 'No CRP MLE training'
    if args.config_crf_u_pp_training > 0:
        print 'CRF Unary and PP combined training'
    if args.cnn_case_sensitivity > 0:
        print 'CNN Upper and Lower Case Sensitivity On'
    else:
        print 'CNN Upper and Lower Case Sensitivity Off'
    if args.config_pp_highway_network > 0:
        print 'PP Highway Network On'
    else:
        print 'PP Highway Network Off'
    if args.config_layer_normalization > 0:
        print 'config_layer_normalization On'
    else:
        print 'config_layer_normalization Off'
    if args.w2v_training > 0:
        print 'Fine-tune W2V On'
    else:
        print 'Fine-tune W2V Off'
        
def main(args):
    print_configuration(args)
    data_path = '../data/'
    datasets = ['BC2GM-IOBES/', 'BC4CHEMD-IOBES/', 'BC5CDR-IOBES/', 'BC5CDR-chem-IOBES/', 'BC5CDR-disease-IOBES/', 'JNLPBA-IOBES/', 'NCBI-disease-IOBES/', 'Ex-PTM-IOBES/', 'linnaeus-IOBES/'] 
    file_list = ['train.tsv', 'devel.tsv', 'test.tsv']
    files = [['train.tsv', 'devel.tsv'], 'test.tsv']
    
    w2v_model_path = '../w2v/'
    w2v_file = 'wikipedia-pubmed-and-PMC-w2v.bin'
    vocab_file = 'pickle/vocab.pkl'
    pos_idx_file = 'pickle/pos2idx_' + datasets[args.dataset][:-1] + '.pkl' 
    
    reader = loader.Loader()
    train_dataset = DatasetPreprocessing(data_path, datasets[args.dataset], files[0])
    
    if not os.path.exists(vocab_file):
        all_sentences = []
        for dataset in datasets:
            for file in file_list:
                sentences, labels = reader.gathering_all_words(data_path, dataset, file)
                reader.write_all_NERs(data_path, dataset, file)
                print len(sentences)
                all_sentences.append(sentences)
    
        vocab = create_vocab(all_sentences)
        pickle.dump(vocab, open(vocab_file, 'wb'), protocol=2)
    else:
        vocab = pickle.load(open(vocab_file, 'rb'))
    
    print '# of vocabulary = {}'.format(len(vocab))
    
    word2idx_file = 'pickle/word2idx.pkl'
    idx2word_file = 'pickle/idx2word.pkl'
    word2vec_matrix_file = 'pickle/word2vec_matrix.pkl'
    
    if not os.path.exists(word2idx_file) or not os.path.exists(word2idx_file) or not os.path.exists(word2idx_file): 
        w2v_loader = loadw2v.LoadW2V()
        w2v_loader()
    word2idx = pickle.load(open(word2idx_file, 'rb'))
    idx2word = pickle.load(open(idx2word_file, 'rb'))
    word2vec_matrix = pickle.load(open(word2vec_matrix_file, 'rb'))
    
    assert(len(word2idx) == len(idx2word) == word2vec_matrix.shape[0])

    print 'Total length of word2idx = ', len(word2idx)
    if args.dataset == 0:
        print 'BC2GM Dataset'
        MAX_SENTENCE_LEN = 267
    elif args.dataset == 1:
        print 'BC4CHEMD Dataset'
        MAX_SENTENCE_LEN = 475
    elif args.dataset == 2:
        print 'BC5CDR Dataset'
        MAX_SENTENCE_LEN = 300
    elif args.dataset == 3:
        print 'BC5CDR chemical Dataset'
        MAX_SENTENCE_LEN = 300
    elif args.dataset == 4:
        print 'BC5CDR disease Dataset'
        MAX_SENTENCE_LEN = 300
    elif args.dataset == 5:
        print 'JNLPBA Dataset'
        MAX_SENTENCE_LEN = 220
    elif args.dataset == 6:
        print 'NCBI-disease Dataset'
        MAX_SENTENCE_LEN = 130
    elif args.dataset == 7:
        print 'Ex-PTM Dataset'
        MAX_SENTENCE_LEN = 175
    elif args.dataset == 8:
        print 'linnaeus Dataset'
        MAX_SENTENCE_LEN = 328

    MAX_WORD_IDX = len(word2idx) + 1
    training_sentences = []
    training_sentences_lengths = []
    training_labels = []

    if os.path.exists(pos_idx_file):
        pos_exist = True
    else:
        pos_tags = set()
        pos_exist = False
        
    nlp = spacy.load('en_core_web_sm')
    read_characters = character.CharacterProcessing()
    total_sentences = []
    for file in file_list:
        sentences = []
        with open(data_path + datasets[args.dataset] + file) as fin:
            reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            sentence = []
            for i, row in enumerate(reader):
                if len(row) < 2:
                    if pos_exist == False:
                        _s = ' '.join(sentence)
                        doc = nlp(_s.decode('utf-8'))
                        pos_sentence = []
                        for token in doc:
#                             pos_sentence.append(token.pos_.encode('ascii', 'ignore'))
                            pos_sentence.append(token.tag_.encode('ascii', 'ignore'))
                        pos_tags.update(pos_sentence)
                    sentences.append(sentence)
                    sentence = []
                else:
                    token = row[0]
                    sentence.append(token)
        total_sentences.append(sentences)
    
    if pos_exist == False:
        print pos_tags, len(pos_tags)
        
    if os.path.exists(pos_idx_file):
        pos2idx = pickle.load(open(pos_idx_file, 'rb'))
        for tag, idx in pos2idx.items():
            print tag + ' : ' + str(idx) 
    else:
        pos2idx = dict()
        for i, pos in enumerate(pos_tags):
            pos2idx[pos] = i
        print pos + ' : ' + str(i)
        pickle.dump(pos2idx, open(pos_idx_file, 'wb'), protocol=2)
    
    case_sensitivity = args.cnn_case_sensitivity
    if case_sensitivity == True:
        print 'Character Case Sensitivity On'
        if args.character_model == 'lstm':
            print 'LSTM character set'
            n_characters, start_char_idx, end_char_idx = read_characters.build_train_characters_with_case_sensitivity_LSTM(total_sentences, MAX_SENTENCE_LEN)
        else:
            print 'CNN character set'
            n_characters, start_char_idx, end_char_idx = read_characters.build_train_characters_with_case_sensitivity_for_CNN(total_sentences, MAX_SENTENCE_LEN)
    else:
        print 'Character Case Sensitivity Off'
        if args.character_model == 'lstm':
            print 'LSTM character set'
            n_characters, start_char_idx, end_char_idx = read_characters.build_train_characters_LSTM(total_sentences, MAX_SENTENCE_LEN)
        else:
            n_characters, start_char_idx, end_char_idx = read_characters.build_train_characters(total_sentences, MAX_SENTENCE_LEN)
    print('n characters = {}'.format(n_characters))
    print('Start Char index = {}'.format(start_char_idx))
    print('End Char index = {}'.format(end_char_idx))
    char_training_sentences = []
    char_training_lengths = []
    training_pos_sentences = []
    
#     for file in files:
    _sentences, _labels = create_testing_sentences(data_path, datasets[args.dataset], files[1])
    sentences, lengths, labels, pos_sentences = create_training_samples(args.dataset, _sentences, _labels, word2idx, nlp, pos2idx, MAX_SENTENCE_LEN, MAX_WORD_IDX)
    training_pos_sentences.append(pos_sentences)
    
    print 'Testing Dataset Done'
#         sentences, lengths, labels = create_binary_training_samples(_sentences, _labels, word2idx, MAX_SENTENCE_LEN, MAX_WORD_IDX)
    training_sentences.append(sentences)
    training_sentences_lengths.append(lengths)
    training_labels.append(labels)
    if case_sensitivity == True:
        if args.character_model == 'lstm':
            print 'LSTM Case sensitive dataset'               
            train_character_dataset, char_lengths = read_characters.build_train_characters_dataset_with_case_sensitivity_for_LSTM(_sentences)
        else:
#             train_character_dataset = read_characters.build_train_characters_dataset_with_case_sensitivity_for_CNN(training_sentences[i])
            train_character_dataset, char_lengths = read_characters.build_train_characters_dataset_with_case_sensitivity_for_CNN(_sentences)
        print('length of sentence in character dataset = {}'.format(len(train_character_dataset)))
        char_training_sentences.append(train_character_dataset)
        char_training_lengths.append(char_lengths)
    else:
        if args.character_model == 'lstm':
            train_character_dataset, char_lengths = read_characters.build_train_characters_dataset_LSTM(_sentences)
        else:
            train_character_dataset, char_lengths = read_characters.build_train_characters_dataset(_sentences)
        print('length of sentence in character dataset = {}'.format(len(train_character_dataset)))
        char_training_sentences.append(train_character_dataset)
        char_training_lengths.append(char_lengths)

    directory = '../models'
    param_names = ['U', 'W', 'V']
    learning_mode = 'pure'

    max_token_length = read_characters.max_token_length
    character_embedding_dim = 15
    word_embedding_dim = 200
    pos_embedding_dim = 10
    n_pos = len(pos2idx)
#     n_filter = [100, 200, 200, 150, 150, 150, 150]
#     n_filter = [80, 200, 200, 200, 200, 200, 200]
    n_filter = [80, 200, 200, 200, 200, 200] 
#     n_filter = [80, 100, 150, 150, 150] 
    n_width = [1,2,3,4,5,6]
#     n_width = [1,2,3,4,5,6,7]
    cnn_output_dim = np.sum(n_filter) # 780
    input_mlp_dim = args.input_mlp_dim  
    lstm_hidden_dim = args.lstm_hidden_dim      #
    lstm_hidden_dim_2 = args.lstm2_hidden_dim   #
    if args.dataset == 0 or args.dataset == 1 or args.dataset == 3 or args.dataset == 4 or args.dataset == 6 or args.dataset == 7 or args.dataset == 8:
        output_dim = 5
    elif args.dataset == 2:
        output_dim = 9
    elif args.dataset == 5:
        output_dim = 21
    else:
        output_dim = 9
    mlp_dim = args.mlp_dim              #
    loss_f = 'max_margin'
    gradient_threshold = args.gradient_threshold
    l2_coef = args.weight_decay_rate
    weight_decay = args.weight_decay
    config_gru = args.config_gru
    config_bilstm_2_layer = args.bilstm_2_added   #
    config_mlp_layer = args.final_mlp_added
    config_cnn_shared = args.shared_cnn
    cnn_character_embedding = True if args.character_model == 'cnn' else False
    w2v_included = True
    w2v_training = args.w2v_training
    optimization = args.optimization 
    dropout_on = args.dropout_on
    dropout_p_x = args.dropout_rate_x
    dropout_p_s = args.dropout_rate_s
    pairwise_potential_training = args.pairwise_potential  #
    attention_in_PP = False
    pairwise_encoder_decoder_training = args.pairwise_potential_encoder_decoder
    separate_potential = args.separate_pairwise_potential   #
    config_lstm_character = True if args.character_model == 'lstm' else False
    highway_network = False
    lstm_hidden_char_dim = args.lstm_char_hidden_dim
    pos_enable = True
    with_forward_backward = args.forward_backward  #
    sentence_level_phrase_transition = args.sentence_level_phrase_transition
    sentence_level_type_transition = args.sentence_level_type_transition
    pp_type_sentence_embedding_status = args.pp_type_sentence_embedding_status
    pp_output_dim = 5
    pp_character_cnn_enable = True
    pp_w2v_enable = args.w2v_pairwise_potential     #
    lstm_pp_hidden_dim = args.lstm_pp_hidden_dim   #
    pp_mlp_dim = args.pp_mlp_dim          #
    pp_mlp_dim2 = args.pp_mlp_dim2
    pp_decoder_hiddem_dim = args.lstm_pp_decoder_hidden_dim  #
    pp_att_scoring_mlp_dim = 512
    lstm_pp_type_hidden_dim = args.lstm_pp_type_hidden_dim  #
    pp_type_output_dim = 3
    pp_type_mlp_dim = args.pp_type_mlp_dim     #
    pp_type_decoder_hiddem_dim = args.lstm_pp_type_decoder_hidden_dim  #
    orthogonal_weight_init = args.orthogonal_weight_init
    exclusive_begin_end_tagging = args.exclusive_begin_end_tagging
    config_input_mlp_layer = args.input_mlp
    config_initial_hidden_modeling = args.config_initial_hidden_modeling
    config_input_mlp_activation = args.input_mlp_activiation
    config_pairwise_output_modeling = args.pairwise_output
    config_pp_joint_training = args.config_pp_joint_training
    config_crf_pp_training = args.config_crf_pp_training
    config_no_crf_training = args.config_no_crf_training
    config_crf_u_pp_training = args.config_crf_u_pp_training
    config_pp_highway_network = args.config_pp_highway_network
    config_layer_normalization = args.config_layer_normalization 
    
    print 'L2 coefficient = ', l2_coef
        
    testing_samples = np.array(training_sentences[0])
    testing_lengths = np.array(training_sentences_lengths[0])
    testing_targets = np.array(training_labels[0])
    testing_chars = np.array(char_training_sentences[0])
    testing_char_lengths = np.array(char_training_lengths[0])
    testing_pos = np.array(training_pos_sentences[0])
  
    print('test sample shape = {}'.format(testing_samples.shape))
    print('test length shape = {}'.format(testing_lengths.shape))
    print('test targets shape = {}'.format(testing_targets.shape))
    print('test char shape = {}'.format(testing_chars.shape))
    print('test char length shape = {}'.format(testing_char_lengths.shape))
    print('test pos shape = {}'.format(testing_pos.shape))

    print('word2vec matrix shape = {}'.format(word2vec_matrix.shape))
    
    training_samples, training_lengths, training_targets, training_char_samples, _, training_pos = random_shuffle_train_dataset(train_dataset, 
                                                                                                                                args.dataset,
                                                                                                                                word2idx, 
                                                                                                                                nlp, 
                                                                                                                                pos2idx, 
                                                                                                                                read_characters, 
                                                                                                                                case_sensitivity, 
                                                                                                                                config_lstm_character, 
                                                                                                                                MAX_SENTENCE_LEN, 
                                                                                                                                MAX_WORD_IDX)

    training_samples = np.array(training_samples)
    training_lengths = np.array(training_lengths)
    training_targets = np.array(training_targets)
    training_char_samples = np.array(training_char_samples)
    training_pos = np.array(training_pos)
        
    print('train sample shape = {}'.format(training_samples.shape))
    print('train length shape = {}'.format(training_lengths.shape))
    print('train targets shape = {}'.format(training_targets.shape))
    print('train char shape = {}'.format(training_char_samples.shape))
    print('train pos shape = {}'.format(training_pos.shape))
    
    model_factory = initialize.InitializeModel(optimization,
                                  orthogonal_weight_init,
                                  word2vec_matrix,
                                  word_embedding_dim,
                                  n_characters, 
                                  character_embedding_dim,
                                  n_pos, 
                                  pos_embedding_dim,
                                  max_token_length,
                                  n_filter,
                                  n_width,
                                  cnn_output_dim,
                                  lstm_hidden_dim_2,
                                  lstm_hidden_dim,
                                  input_mlp_dim, 
                                  output_dim, 
                                  mlp_dim,
                                  training_samples,
                                  training_lengths,
                                  training_targets,
                                  training_char_samples,
                                  training_pos,
                                  testing_samples,
                                  testing_lengths,
                                  testing_targets,
                                  testing_chars,
                                  testing_char_lengths,
                                  testing_pos,
                                  dropout_on,
                                  dropout_p_x,
                                  dropout_p_s,
                                  config_layer_normalization,
                                  args.dataset,
                                  exclusive_begin_end_tagging,
                                  loss_f,
                                  weight_decay,
                                  gradient_threshold,
                                  l2_coef,
                                  cnn_character_embedding,
                                  w2v_included,
                                  config_input_mlp_layer,
                                  config_initial_hidden_modeling,
                                  config_pp_joint_training,
                                  config_crf_pp_training,
                                  config_pp_highway_network,
                                  config_no_crf_training,
                                  config_crf_u_pp_training,
                                  config_input_mlp_activation,
                                  config_pairwise_output_modeling,
                                  config_cnn_shared,
                                  config_mlp_layer,
                                  config_bilstm_2_layer,
                                  w2v_training,
                                  pp_mlp_dim,
                                  pp_mlp_dim2,
                                  pp_decoder_hiddem_dim,
                                  pp_att_scoring_mlp_dim,
                                  pairwise_potential_training, 
                                  attention_in_PP,
                                  pairwise_encoder_decoder_training,
                                  config_gru,
                                  config_lstm_character,
                                  lstm_hidden_char_dim,
                                  separate_potential,
                                  highway_network,
                                  pos_enable,
                                  with_forward_backward,
                                  sentence_level_phrase_transition,
                                  sentence_level_type_transition,
                                  pp_type_sentence_embedding_status,
                                  lstm_pp_hidden_dim,
                                  pp_output_dim,
                                  pp_character_cnn_enable,
                                  pp_w2v_enable,
                                  lstm_pp_type_hidden_dim,
                                  pp_type_output_dim,
                                  pp_type_mlp_dim,
                                  pp_type_decoder_hiddem_dim
                                  )

    model = model_factory.getModel()
    
    print('Theano Graph Generated...')

    n_epoch = 100
    n_test_samples = testing_samples.shape[0]
    n_minibatch = args.minibatch_size
    n_display_samples_train = 10
    n_display_samples_test = 100
    
    if args.dataset == 0:
        label_dict = {0:'O', 1:'B-GENE', 2:'I-GENE', 3:'E-GENE', 4:'S-GENE'}
    elif args.dataset == 1:
        label_dict = {0:'O', 1:'B-Chemical', 2:'I-Chemical', 3:'E-Chemical', 4:'S-Chemical'}
    elif args.dataset == 2:
        label_dict = {0:'O', 1:'B-Disease', 2:'I-Disease', 3:'E-Disease', 4:'S-Disease', 5:'B-Chemical', 6:'I-Chemical', 7:'E-Chemical', 8:'S-Chemical'}
    elif args.dataset == 3:
        label_dict = {0:'O', 1:'B-Chemical', 2:'I-Chemical', 3:'E-Chemical', 4:'S-Chemical'}
    elif args.dataset == 4:
        label_dict = {0:'O', 1:'B-Disease', 2:'I-Disease', 3:'E-Disease', 4:'S-Disease'}
    elif args.dataset == 5:
        label_dict = {0:'O', 1:'B-protein', 2:'I-protein', 3:'E-protein', 4:'S-protein', 5:'B-cell_type', 6:'I-cell_type', 7:'E-cell_type', 8:'S-cell_type', 9:'B-cell_line', 10:'I-cell_line', 11:'E-cell_line', 12:'S-cell_line', 13:'B-DNA', 14:'I-DNA', 15:'E-DNA', 16:'S-DNA', 17:'B-RNA', 18:'I-RNA', 19:'E-RNA', 20:'S-RNA'}
    elif args.dataset == 6:
        label_dict = {0:'O', 1:'B-Disease', 2:'I-Disease', 3:'E-Disease', 4:'S-Disease'}
    elif args.dataset == 7:
        label_dict = {0:'O', 1:'B-Protein', 2:'I-Protein', 3:'E-Protein', 4:'S-Protein'}
    elif args.dataset == 8:
        label_dict = {0:'O', 1:'B-Species', 2:'I-Species', 3:'E-Species', 4:'S-Species'}

    w2v_joint_training_epoch = 0
    test_start_epoch = args.test_start
    log_file_name = 'crf_train_'
    log_file_name += model.getName();
    log_file_name += 'dataset_' + str(args.dataset)
    log_file_name += '_bilstm' + str(lstm_hidden_dim)
    log_file_name += '_cnn_' + '_'.join([str(i) for i in n_filter]) 
    log_file_name += '_' + time.strftime('%H', time.localtime()) + '_hour'
    log_file_name += '.txt'
    with open(log_file_name, 'w', 0) as f_log:
        f_log.write('Theano Graph Generated\n')
        f_log.write('Training Start...\n')
        best_i = 0
        best_recall = 0. 
        best_precision = 0.
        best_f_score = 0.

        for i in xrange(0, n_epoch):
            acc_loss = 0.
            acc_loss_u = 0.
            acc_loss_pp = 0.
            total_loss = 0.
            total_loss_u = 0.
            total_loss_pp = 0.
            acc_all_gnorm = 0.
            acc_pp_gnorm = 0.
            sentence_cnn_statistics = []
#             sentence_dF_statistics = []
            acc_dP = 0.
#             acc_dF = 0.
#             if i > 0:
#             training_samples, training_lengths, training_targets, training_char_samples, training_char_lengths, training_pos = random_shuffle_train_dataset(train_dataset, args.dataset,
#                                                                                                                                                             word2idx, nlp, pos2idx, read_characters, case_sensitivity, config_lstm_character, MAX_SENTENCE_LEN, MAX_WORD_IDX)
            n_training_samples = len(training_samples)
            training_indexes = np.random.permutation(n_training_samples)
#             training_indexes = np.arange(n_training_samples)
            if n_training_samples % n_minibatch == 0:
                n_set_minibatch = n_training_samples / n_minibatch
            else:
                n_set_minibatch = (n_training_samples / n_minibatch) + 1
            for j in xrange(n_set_minibatch):
                if j == (n_set_minibatch-1):
                    minibatch_training_indexes = np.array(training_indexes[j*n_minibatch:])
                else:
                    minibatch_training_indexes = np.array(training_indexes[j*n_minibatch:(j+1)*n_minibatch])
                
                if config_lstm_character == True:
                    loss, all_gnorm, pp_norm, dP = model.train(minibatch_training_samples, minibatch_training_lengths, minibatch_training_char_samples, minibatch_training_char_lengths, minibatch_training_pos, minibatch_training_targets)
                else:
#                     print minibatch_training_indexes
                    loss, all_gnorm, pp_norm, loss_u, loss_pp, dP = model.train(minibatch_training_indexes)
#                     [loss, Z_u, Z_pp, Z_lop, dLoss_dP], 
#                     loss, all_gnorm, pp_norm, sentence_cnn, dP = model.train(minibatch_training_samples, minibatch_training_lengths, minibatch_training_char_samples, minibatch_training_pos, minibatch_training_targets)
#                     loss = model.train(minibatch_training_samples, minibatch_training_lengths, minibatch_training_pos, minibatch_training_targets, 1, 1)
#                 if config_lstm_character == True:
#                     pass
#                 else:
#                     sentence_cnn_statistics.append(sentence_cnn)
#                 sentence_dF_statistics.append(dLoss_dF)
#                 for i, dF in dloss_dF:
#                     print i, dF
#                 print 'real path = ', loss_u
#                 print 'partition = ', loss_pp
#                 print 'dropout sequence = ', dropout_sequence
#                 print 'another sequence = ', another_sequence
                acc_loss += loss
                acc_loss_u += loss_u
                acc_loss_pp += loss_pp
                acc_all_gnorm += all_gnorm
                acc_pp_gnorm += pp_norm
                acc_dP += dP
#                 acc_dF += np.sum(dLoss_dF)
                if j > 0 and j % n_display_samples_train == 0:
                    training_status_log =  'In {}, {}:{}/{},Error:{:06.5f},U_Error:{:05.4f},PP_Error:{:05.4f},Norm:{:06.5f},PP_Norm:{:06.5f},dP:{:05.4f}\n'.format(
                        i,
                        'Train',
                        j, 
                        n_training_samples/n_minibatch, 
                        acc_loss/n_display_samples_train,
                        acc_loss_u/n_display_samples_train,
                        acc_loss_pp/n_display_samples_train,
                        acc_all_gnorm/n_display_samples_train,
                        acc_pp_gnorm/n_display_samples_train,
                        acc_dP/n_display_samples_train)
                    print training_status_log
                    f_log.write(training_status_log)
                    total_loss += acc_loss
                    total_loss_u += acc_loss_u
                    total_loss_pp += acc_loss_pp
                    acc_loss = 0.
                    acc_loss_u = 0.
                    acc_loss_pp = 0.
                    acc_all_gnorm = 0.
                    acc_pp_gnorm = 0.
                    acc_dP = 0.

            total_loss /= n_set_minibatch
            total_loss_u /= n_set_minibatch
            total_loss_pp /= n_set_minibatch
            train_epoch_log = 'In {}, {} Error:{:06.5f}, U_Error:{:06.5f}, PP_Error:{:06.5f}\n'.format(
                i,
                'Train',
                total_loss,
                total_loss_u,
                total_loss_pp) 
            print train_epoch_log
            f_log.write(train_epoch_log)
#             cnn_statistics = np.transpose(np.percentile(np.array(sentence_cnn_statistics), [0,25,50,75,100], axis=0))
#             F_statistics = np.mean(np.array(sentence_dF_statistics), axis=0)
#             for i_th, each_cnn in enumerate(cnn_statistics):
#                 print i_th, 'Activation : ', each_cnn

#             model.replace_train_dataset(train_words, train_lengths, train_targets, train_char_samples)
            
            if i >= test_start_epoch:
                evaluate_token_list = []
                for j in xrange(n_test_samples):
#                     n_recall, n_precision, n_true_positives, prediction = model.testing(0, j)
                    prediction, unary_score, pp_score = model.testing(j)
                    test_sentence = training_sentences[0][j][:training_sentences_lengths[0][j]]
                    test_target = training_labels[0][j][:training_sentences_lengths[0][j]]
                    sentence_words = [idx2word[word_id] for word_id in test_sentence if word_id != MAX_WORD_IDX]
                    assert len(sentence_words) == len(prediction) == len(test_target) == len(unary_score)
                    pp_score_print = False if type(pp_score[0]) == np.float32 else True
                    idx_pp = 0
                    for word, word_label, target, score in zip(sentence_words, prediction, test_target, unary_score):
                        if args.logging_label_prediction > 0:
                            scores = '_'.join([str(round(_s,2)) for _s in score]) 
                            word_log_str = 'In {} : {} : {} , {}, {}\n'.format(str(i), word, label_dict[word_label], label_dict[target], scores)
                            f_log.write(word_log_str)
#                             if idx_pp < len(unary_score)-1 and pp_score_print: 
#                                 _pp_scores = pp_score[idx_pp]
#                                 pp_scores = ''
#                                 for _i, _pp_score in enumerate(_pp_scores):
#                                     if _i != 0 and _i % output_dim == 0:
#                                         pp_scores += '\n'
#                                     pp_scores += str(round(_pp_score, 2))
#                                     pp_scores += ' '
# #                                 pp_scores = ' '.join([str(round(_pp,2)) for _pp in pp_score[idx_pp]]) 
#                                 pp_score_log_str = '{}\n'.format(pp_scores)
#                                 f_log.write(pp_score_log_str)
#                                 idx_pp += 1 
                        evaluate_token_str = '{} {} {}'.format(word, label_dict[target], label_dict[word_label])
                        evaluate_token_list.append(evaluate_token_str)
#                         print word_log_str
#                     print log_str
    #                     print 'sentence = ', sentence_words
    #                     print 'prediction = ', prediction
#                     n_recall, n_precision, n_true_positives = evaluate(prediction, label) 
                    
#                     log_str = 'n_recall : {}, n_precision : {}, n_trues : {}\n'.format(n_recall, n_precision, n_true_positives)
#                     f_log.write(log_str)

#                     if j > 0 and j % n_display_samples_test == 0:
#                         local_recall = local_n_trues / local_n_recall if local_n_recall != 0 else 0
#                         local_precision = local_n_trues / local_n_precision if local_n_precision != 0 else 0
#                         f_score = 2 * (local_recall * local_precision) / (local_recall + local_precision) if (local_recall + local_precision) != 0 else 0
#                         training_status_log =  'In {}, {}:{}/{},F-Score:{:05.4f},Recall:{:05.4f},Precision:{:05.4f}\n'.format(
#                             i,
#                             'Test',
#                             j, 
#                             n_test_samples, 
#                             f_score, 
#                             local_recall, 
#                             local_precision)
#                         print training_status_log
#                         f_log.write(training_status_log)
#                         local_n_recall = 0.
#                         local_n_precision = 0.
#                         local_n_trues = 0
#                
#                 total_recall = total_n_trues / total_n_recall
#                 total_precision = total_n_trues / total_n_precision
#                 total_f_score = 2 * (total_recall * total_precision) / (total_recall + total_precision)
                counts = conlleval.evaluate(evaluate_token_list)
                total_precision, total_recall, total_f_score = conlleval.report(counts, f_log)  # overall.prec, overall.rec, overall.fscore
                
                [best_f_score, best_recall, best_precision, best_i] = [total_f_score, total_recall, total_precision, i] if total_f_score > best_f_score else [best_f_score, best_recall, best_precision, best_i]
                best_log = 'Best In {}, {} Recall:{:05.4f},Precision:{:05.4f},F-Score:{:05.4f}\n'.format(
                    best_i,
                    'Test',
                    best_recall, 
                    best_precision, 
                    best_f_score)
                print best_log
                f_log.write(best_log)
            
                if best_i == i:
                    transition_matrix = model.get_parameters()
                    for i, row in enumerate(transition_matrix):
                        print i, row
                    param_name = 'transition_matrix_'
                    param_name += 'dataset_' + str(args.dataset) + '_' 
                    param_name += str(cnn_output_dim) + '_'
                    param_name += str(lstm_hidden_dim)
                    param_names = [param_name]
                    params = [transition_matrix]
                    directory = '../models'
                    for suffix, param in zip(param_names, params):
                        with open('{}/{}.pkl'.format(directory, suffix), 'wb') as fout:
                            pickle.dump(param, fout)    

if __name__=="__main__":
    parser = argparse.ArgumentParser('Learning with Pairwise Potential Neural Networks')
    parser.add_argument('--pairwise_potential', type=int, default='1')
    parser.add_argument('--separate_pairwise_potential', type=int, default='0')
    parser.add_argument('--pairwise_potential_encoder_decoder', type=int, default='1')
    parser.add_argument('--w2v_pairwise_potential', type=int, default='1')
    parser.add_argument('--forward_backward', type=int, default='0')
    parser.add_argument('--sentence_level_phrase_transition', type=int, default='1')
    parser.add_argument('--sentence_level_type_transition', type=int, default='1')
    parser.add_argument('--pp_type_sentence_embedding_status', type=int, default='2')
    parser.add_argument('--bilstm_2_added', type=int, default='0')
    parser.add_argument('--orthogonal_weight_init', type=int, default='0')
    parser.add_argument('--lstm_char_hidden_dim', type=int, default='100')
    parser.add_argument('--input_mlp_dim', type=int, default='600')
    parser.add_argument('--lstm_hidden_dim', type=int, default='800')
    parser.add_argument('--lstm2_hidden_dim', type=int, default='800')
    parser.add_argument('--lstm_pp_hidden_dim', type=int, default='550')
    parser.add_argument('--lstm_pp_type_hidden_dim', type=int, default='550')
    parser.add_argument('--lstm_pp_decoder_hidden_dim', type=int, default='300')
    parser.add_argument('--lstm_pp_type_decoder_hidden_dim', type=int, default='300')
    parser.add_argument('--mlp_dim', type=int, default='512')
    parser.add_argument('--pp_mlp_dim', type=int, default='1024')
    parser.add_argument('--pp_mlp_dim2', type=int, default='256')
    parser.add_argument('--pp_type_mlp_dim', type=int, default='256')
    parser.add_argument('--dropout_on', type=int, default='0')
    parser.add_argument('--dropout_rate_x', type=float, default='0.25')
    parser.add_argument('--dropout_rate_s', type=float, default='0.3')
    parser.add_argument('--weight_decay', type=int, default='0')
    parser.add_argument('--weight_decay_rate', type=float, default='0.00001')
    parser.add_argument('--exclusive_begin_end_tagging', type=int, default='1')
    parser.add_argument('--logging_label_prediction', type=int, default='0')
    parser.add_argument('--minibatch_size', type=int, default='10')
    parser.add_argument('--optimization', type=str, default='adam')
    parser.add_argument('--test_start', type=int, default='0')
    parser.add_argument('--method', type=str, default='new')
    parser.add_argument('--dataset', type=int, default='2')
    parser.add_argument('--character_model', type=str, default='cnn')
    parser.add_argument('--gradient_threshold', type=float, default='10.0')
    parser.add_argument('--final_mlp_added', type=int, default='0')
    parser.add_argument('--shared_cnn', type=int, default='0')
    parser.add_argument('--config_gru', type=int, default='0')
    parser.add_argument('--input_mlp', type=int, default='0')
    parser.add_argument('--input_mlp_activiation', type=str, default='tanh')
    parser.add_argument('--pairwise_output', type=int, default='0')
    parser.add_argument('--config_initial_hidden_modeling', type=int, default='1')
    parser.add_argument('--config_pp_joint_training', type=int, default='0')
    parser.add_argument('--config_crf_pp_training', type=int, default='0')
    parser.add_argument('--config_no_crf_training', type=int, default='0')
    parser.add_argument('--config_crf_u_pp_training', type=int, default='0')
    parser.add_argument('--cnn_case_sensitivity', type=int, default='1')
    parser.add_argument('--config_pp_highway_network', type=int, default='1')
    parser.add_argument('--config_layer_normalization', type=int, default='0')
    parser.add_argument('--w2v_training', type=int, default='0')
    args = parser.parse_args()

    main(args)

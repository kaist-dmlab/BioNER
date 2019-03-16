import gensim
import pickle
import collections
import numpy as np

class LoadW2V:
    def __init__(self):
        self.model_filepath = '../w2v/wikipedia-pubmed-and-PMC-w2v.bin'
    
    def __call__(self):
#     model = gensim.models.word2vec.Word2Vec.load_word2vec_format(model_filepath, binary=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_filepath, binary=True)
        word2vec = model.wv
        del model
        
        vocab_file = 'pickle/vocab.pkl'
        vocab = pickle.load(open(vocab_file, 'rb'))
        print('# of vocabulary = {}'.format(len(vocab)))
         
        unknown_token = '<UNK>'
        word_vector_matrix = []
        word2idx = collections.OrderedDict()
        idx2word = collections.OrderedDict()
        word2idx[unknown_token] = 0
        idx2word[0] = unknown_token
        unknown_vector = np.random.uniform(-0.00001, 0.00001, (200,))
#         print('vector of unknown token = {}.format(unknown_vector))
        word_vector_matrix.append(unknown_vector)
        counter = 0
        idx = 1
        for w in vocab:
            try:
                vec = word2vec[w]
            except KeyError:
                counter += 1
                continue
            word_vector_matrix.append(vec)
            idx2word[idx] = w
            word2idx[w] = idx
            idx += 1
     
        print('n key error counter = ', counter)
        print('n word vector matrix = ', len(word_vector_matrix))
        print('n words not in word2vec = ', counter)
        print('n final vocab = ', idx)
        set_word_map = set(word2idx.keys())
        assert counter == len(vocab.difference(set_word_map))
         
        word2vec_matrix = np.array(word_vector_matrix)
        print('size of word2vec matrix = {}'.format(word2vec_matrix.shape))
        assert(word2vec_matrix.shape[0] == len(word2idx) == len(idx2word))
         
        pickle.dump(word2vec_matrix, open('pickle/word2vec_matrix.pkl', 'wb'), protocol=2)
        pickle.dump(word2idx, open('pickle/word2idx.pkl', 'wb'), protocol=2)
        pickle.dump(idx2word, open('pickle/idx2word.pkl', 'wb'), protocol=2)

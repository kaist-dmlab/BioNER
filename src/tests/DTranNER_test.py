import unittest
import dataset
import torch
import torch.nn as nn
import numpy as np
import sys
from models.DTranNER import DTranNER

class TestNeuralModel(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        tag_to_ix = {'<START>':0, '<STOP>':1, 'Dummy':2}
        character_vocab_size = 10
        word_embedding_dim = 100
        hidden_dim = 100
        pp_hidden_dim = 100
        dropout = False
        filter_num_width = {1:60, 2:200, 3:200, 4:200, 5:200, 6:200}
        self.model = DTranNER(device, tag_to_ix, character_vocab_size, word_embedding_dim, hidden_dim, pp_hidden_dim, dropout, filter_num_width)

    def test_PosEmbeddingShouldHaveOrthogonality(self):
        self.assertTrue(np.allclose(torch.sum(torch.mul(self.model.pos_embedding[0], 
                                                        self.model.pos_embedding[1])).cpu().numpy(), 0., rtol=1e-05, atol=1e-05))

        self.assertTrue(np.allclose(torch.sum(torch.mul(self.model.pos_embedding[1], 
                                                        self.model.pos_embedding[2])).cpu().numpy(), 0., rtol=1e-05, atol=1e-05))

        self.assertTrue(np.allclose(torch.sum(torch.mul(self.model.pos_embedding[2], 
                                                        self.model.pos_embedding[3])).cpu().numpy(), 0., rtol=1e-05, atol=1e-05))

    def test_CheckAutoGradStatusOfPosEmbeddingShouldBeFalse(self):
        for name, param in self.model.named_parameters():
            if name == 'pos_embedding':
                print("requires_grad = %r " %param.requires_grad)
                self.assertFalse(param.requires_grad)

    def tearDown(self):
        pass

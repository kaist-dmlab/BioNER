import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import utils
import numpy as np
from dataset import DatasetPreprosessed
from allennlp.modules.elmo import Elmo

ELMO_OPTIONS_FILE = "~/Downloads/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "~/Downloads/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

class CRF_FB(nn.Module):
    def __init__(self, device, tag_to_ix, n_layers, hidden_dim, hidden_dim_pp, char_cnn, n_chars, char_cnn_filters, pairwise_gate, train_type="sequence", 
                 normalization="weight", elmo_dropout_ratio=0., dropout_ratio=0., shared_lstm=False, inp_config="full", pairwise_query_type='mul', bilinear_dim=300, elmo_dim=1024, 
                 attn='multi', all_test=False, gate_bias=-1., monitor=None, logger=None):
        super(CRF_FB, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_dim_pp = hidden_dim_pp
        self.bilinear_dim = bilinear_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.monitor = monitor
        self.embedding_dim = elmo_dim
        self.normalization = normalization
        self.elmo_dropout_ratio = elmo_dropout_ratio
        self.dropout_ratio = dropout_ratio
        self.train_type = train_type.lower()
        self.n_layers = n_layers
        self.char_cnn = char_cnn
        self.pairwise_gate = pairwise_gate
        self.bilinear_inp_dim = self.embedding_dim
        self.bilinear_out_dim = hidden_dim
        self.char_cnn_highway_bias = -1.
        self.query_dim = hidden_dim
        self.attn_dim = hidden_dim
        self.inp_config = inp_config
        self.shared_lstm = shared_lstm
        self.pairwise_query_type = pairwise_query_type
        self.pairwise_bilinear_pooling = True
        self.all_test = all_test
        self.logger = logger
        self.logger.info("Pairwise Type = {}".format(self.pairwise_query_type))
        
        if self.inp_config != "w2v":
            self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, 1, requires_grad=False, dropout=self.elmo_dropout_ratio)
            self.elmo.to(self.device)

        self.act = nn.ELU()

        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        if self.train_type != "no_unary":
            self.logger.info("Unary Config")
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout_ratio, bidirectional=True).to(device=device)

            self.unary_fc = weight_norm(nn.Linear(2*hidden_dim, 2*hidden_dim, bias=True).to(device=device), dim=None)
            self.init_parameters(self.unary_fc, 'relu')

            self.out_dropout_u_fc = nn.Dropout(self.dropout_ratio)
            self.out_dropout_u_skip = nn.Dropout(self.dropout_ratio)
            
            self.hidden2tag = weight_norm(nn.Linear(2*hidden_dim, self.tagset_size).to(device=device), dim=None)
            self.init_parameters(self.hidden2tag, 'linear')

            tran_init = torch.empty(self.tagset_size, self.tagset_size, dtype=torch.float, requires_grad=True)
            torch.nn.init.normal_(tran_init, mean=0.0, std=1.)
            self.transitions = nn.Parameter(tran_init.to(device=device))
            self.transitions.data[:, tag_to_ix[DatasetPreprosessed.__START_TAG__]] = -100.
            self.transitions.data[tag_to_ix[DatasetPreprosessed.__STOP_TAG__], :] = -100.
        
        if self.train_type != "no_pairwise":
            self.logger.info("Pairwise Config")
            if not self.shared_lstm:
                self.logger.info("Separate LSTMs")
                self.lstm_pairwise = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout_ratio, bidirectional=True).to(device=device)
            else:
                self.logger.info("Shared LSTM")
            self.U = weight_norm(nn.Linear(2*self.hidden_dim, self.hidden_dim_pp).to(device=device), dim=None)
            self.init_parameters(self.U, 'relu')
            self.V = weight_norm(nn.Linear(2*self.hidden_dim, self.hidden_dim_pp).to(device=device), dim=None)
            self.init_parameters(self.V, 'relu')
            self.P = weight_norm(nn.Linear(self.hidden_dim_pp, self.bilinear_dim).to(device=device), dim=None)
            self.init_parameters(self.P, 'relu')
            self.pairwise_fc = weight_norm(nn.Linear(self.bilinear_dim, self.bilinear_dim, bias=True).to(device=device), dim=None)
            self.init_parameters(self.pairwise_fc, 'relu')
            self.dropout_p_mul = nn.Dropout(self.dropout_ratio)
            self.out_dropout_p_fc = nn.Dropout(self.dropout_ratio) 
            self.out_dropout_p_skip = nn.Dropout(self.dropout_ratio) 
            self.hidden2tag_pp = weight_norm(nn.Linear(self.bilinear_dim, self.tagset_size**2).to(device=device), dim=None)
            self.init_parameters(self.hidden2tag_pp, 'linear')
            
        self.__start__ = torch.tensor(self.tag_to_ix[DatasetPreprosessed.__START_TAG__], dtype=torch.long).to(device=device)
        self.__stop__ = torch.tensor(self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__], dtype=torch.long).to(device=device)

    def init_parameters(self, sub_module, nonlinearity="relu"):
        nn.init.xavier_uniform_(sub_module.weight, gain=nn.init.calculate_gain(nonlinearity))
        if sub_module.bias is not None:
            nn.init.constant_(sub_module.bias, 0.)

    def init_hidden(self):
        hidden = torch.zeros((2*self.n_layers, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True)
        cell = torch.zeros((2*self.n_layers, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True) 
        return (hidden, cell)
         
    def init_pairwise_hidden(self):
        hidden = torch.zeros((2*self.n_layers, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True)
        cell = torch.zeros((2*self.n_layers, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True) 
        return (hidden, cell)

    def forward_alg_unary(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -100., dtype=torch.float, requires_grad=True).to(device=self.device)
        init_alphas[0][self.__start__] = 0.
 
        forward_var = init_alphas
 
        for i, feat in enumerate(feats):
            alphas_t = []  
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[:,next_tag].view(1, -1)
                assert emit_score.size() == trans_score.size()
                next_tag_var = forward_var + emit_score + trans_score
                alphas_t.append(utils.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[:, self.__stop__].view(1, -1)
        alpha = utils.log_sum_exp(terminal_var)
        return alpha

    def forward_alg_pairwise(self, feats):
        init_alphas = torch.full((1, self.tagset_size), 0, dtype=torch.float, requires_grad=True).to(device=self.device)
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []  
            for next_tag in range(self.tagset_size):
                trans_score = feat.view(self.tagset_size, self.tagset_size)[:, next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score
                alphas_t.append(utils.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var
        alpha = utils.log_sum_exp(terminal_var)
        return alpha

    def score_sentence_unary(self, feats, tags):
        score = torch.tensor(0., dtype=torch.float, requires_grad=True).to(device=self.device)
        tags = tags.squeeze().type(torch.long)
        start = self.__start__.unsqueeze(0)
        if tags.dim() == 0:
            tags = tags.unsqueeze(0)
            
        tags = torch.cat([start, tags])
        for i, feat in enumerate(feats):
            score = score + feat[tags[i+1]] + self.transitions[tags[i], tags[i+1]]
        score = score + self.transitions[tags[-1], self.__stop__]
        return score

    def score_sentence_pairwise(self, feats, tags):
        score = torch.tensor(0., dtype=torch.float, requires_grad=True).to(device=self.device)
        tags = tags.squeeze().type(torch.long)
        start = self.__start__.unsqueeze(0)
        stop = self.__stop__.unsqueeze(0)
        if tags.dim() == 0:
            tags = tags.unsqueeze(0)

        tags = torch.cat([start, tags, stop]).to(device=self.device)
        for i, feat in enumerate(feats):
            score = score + feat.view(self.tagset_size, self.tagset_size)[tags[i], tags[i+1]]
        return score

    def get_unary_lstm_features(self, sentence, iter):
        self.hidden = self.init_hidden()
        embeds = sentence.view(-1, 1, self.embedding_dim)
         
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        return lstm_out 
    
    def get_pairwise_lstm_features(self, sentence, iter):
        self.hidden = self.init_pairwise_hidden()
        embeds = sentence.view(-1, 1, self.embedding_dim)
        
        lstm_out, self.hidden = self.lstm_pairwise(embeds, self.hidden)
        
        return lstm_out
    
    def get_unary_features(self, lstm_out, iter):
        fc_inp = lstm_out[1:-1].squeeze(1)
        fc_out = self.out_dropout_u_fc(self.act(self.unary_fc(fc_inp))) + self.out_dropout_u_skip(fc_inp)
        feats = self.hidden2tag(fc_out)               
        
        return feats

    def get_pairwise_features(self, lstm_out, iter):
        U_inp = lstm_out[:-1].squeeze(1)
        V_inp = lstm_out[1:].squeeze(1)

        if self.pairwise_bilinear_pooling:
            U = self.U(U_inp)
            V = self.V(V_inp)
            h = torch.mul(U, V)
            fc_inp = self.act(self.P(self.dropout_p_mul(h)))
            fc_out = self.out_dropout_p_fc(self.act(self.pairwise_fc(fc_inp))) + self.out_dropout_p_skip(fc_inp)
            feats = self.hidden2tag_pp(fc_out)
        else:
            U = U_inp
            V = V_inp
            fc_inp = torch.cat([U, V, torch.mul(U, V)], 1)
            fc_out = self.out_dropout_p_fc(self.act(self.pairwise_fc(fc_inp))) + self.out_dropout_p_skip(fc_inp)
            feats = self.hidden2tag_pp(fc_out)
            
        return feats

    def viterbi_decode(self, feats, feats_pp):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -100., dtype=torch.float).to(device=self.device)
        init_vvars[0][self.__start__] = 0
  
        forward_var = init_vvars
        for feat, feat_pp in zip(feats, feats_pp[:-1]):
            bptrs_t = []  
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                pairwise_transition_score = feat_pp.view(self.tagset_size, self.tagset_size)[:,next_tag].view(1, -1)
                next_tag_var = forward_var + self.transitions[:, next_tag].view(1, -1) + pairwise_transition_score
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + feats_pp[-1].view(self.tagset_size, self.tagset_size)[:, self.__stop__].view(1,-1)
        best_tag_id = utils.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
  
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def viterbi_decode_unary(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -100., dtype=torch.float).to(device=self.device)
        init_vvars[0][self.__start__] = 0
 
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[:, next_tag].view(1, -1)
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (feat + torch.cat(viterbivars_t)).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[:, self.__stop__].view(1, -1)
        best_tag_id = utils.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
 
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def viterbi_decode_pairwise(self, feats_pp):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -100., dtype=torch.float).to(device=self.device)
        init_vvars[0][self.__start__] = 0
 
        forward_var = init_vvars
        for feat_pp in feats_pp[:-1]:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                pairwise_transition_score = feat_pp.view(self.tagset_size, self.tagset_size)[:,next_tag].view(1, -1)
                next_tag_var = forward_var + pairwise_transition_score
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t)).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + feats_pp[-1].view(self.tagset_size, self.tagset_size)[:, self.__stop__].view(1,-1)
        best_tag_id = utils.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
 
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sequence, words, chars, tags, iter):
        if self.inp_config == "full":
            sequence = self.elmo(sequence)["elmo_representations"][0].squeeze(0)
            inp_feats = torch.cat((sequence, words), 2)
        elif self.inp_config == "w2v":
            inp_feats = words
        else:
            inp_feats = self.elmo(sequence)["elmo_representations"][0].squeeze(0)
        inp_feats = self.layer_norm(inp_feats)
                     
        if self.train_type == "no_unary":
            pairwise_inp_feats = self.get_pairwise_lstm_features(inp_feats, iter)
            pairwise_feats = self.get_pairwise_features(pairwise_inp_feats, iter)
            forward_score_pp = self.forward_alg_pairwise(pairwise_feats)
            gold_score_pp = self.score_sentence_pairwise(pairwise_feats, tags.type(torch.IntTensor).to(device=self.device))
            loss_pairwise = forward_score_pp - gold_score_pp
            loss = loss_pairwise
        
            return loss, loss_pairwise, loss_pairwise
        
        elif self.train_type == "no_pairwise":
            unary_inp_feats = self.get_unary_lstm_features(inp_feats, iter)
            unary_feats = self.get_unary_features(unary_inp_feats, iter)
            forward_score_u = self.forward_alg_unary(unary_feats)
            gold_score_u = self.score_sentence_unary(unary_feats, tags.type(torch.IntTensor).to(device=self.device))
            loss_unary = forward_score_u - gold_score_u
            loss = loss_unary
            
            return loss, loss_unary, loss_unary
        
        else:
            unary_inp_feats = self.get_unary_lstm_features(inp_feats, iter)
            if self.shared_lstm:
                pairwise_inp_feats = unary_inp_feats
            else:
                pairwise_inp_feats = self.get_pairwise_lstm_features(inp_feats, iter)

            unary_feats = self.get_unary_features(unary_inp_feats, iter)
            pairwise_feats = self.get_pairwise_features(pairwise_inp_feats, iter)
            forward_score_u = self.forward_alg_unary(unary_feats)
            gold_score_u = self.score_sentence_unary(unary_feats, tags.type(torch.IntTensor).to(device=self.device))
            loss_unary = forward_score_u - gold_score_u

            forward_score_pp = self.forward_alg_pairwise(pairwise_feats)
            gold_score_pp = self.score_sentence_pairwise(pairwise_feats, tags.type(torch.IntTensor).to(device=self.device))
            loss_pairwise = forward_score_pp - gold_score_pp

            loss = loss_unary + loss_pairwise
            
        return loss, loss_unary, loss_pairwise

    def forward(self, sequence, words, chars):
        if self.inp_config == "full":
            sequence = self.elmo(sequence)["elmo_representations"][0].squeeze(0)
            inp_feats = torch.cat((sequence, words), 2)
        elif self.inp_config == "w2v":
            inp_feats = words
        else:
            inp_feats = self.elmo(sequence)["elmo_representations"][0].squeeze(0)
        
        inp_feats = self.layer_norm(inp_feats)
                  
        if self.train_type == "no_unary":
            pairwise_inp_feats = self.get_pairwise_lstm_features(inp_feats, iter)
            pairwise_feats = self.get_pairwise_features(pairwise_inp_feats, iter)
            score, tag_seq = self.viterbi_decode_pairwise(pairwise_feats)
        elif self.train_type == "no_pairwise":
            unary_inp_feats = self.get_unary_lstm_features(inp_feats, iter)
            unary_feats = self.get_unary_features(unary_inp_feats, iter)
            score, tag_seq = self.viterbi_decode_unary(unary_feats) 
        else:
            unary_inp_feats = self.get_unary_lstm_features(inp_feats, iter)
            if self.shared_lstm:
                pairwise_inp_feats = unary_inp_feats
            else:
                pairwise_inp_feats = self.get_pairwise_lstm_features(inp_feats, iter)

            unary_feats = self.get_unary_features(unary_inp_feats, iter)
            pairwise_feats = self.get_pairwise_features(pairwise_inp_feats, iter)
            score, tag_seq = self.viterbi_decode(unary_feats, pairwise_feats)
            if self.all_test:
                score_u, tag_seq_u = self.viterbi_decode_unary(unary_feats)
                score_p, tag_seq_p = self.viterbi_decode_pairwise(pairwise_feats)
                return score, tag_seq, score_u, tag_seq_u, score_p, tag_seq_p  
        return score, tag_seq

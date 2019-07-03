import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import utils
import numpy as np
from models.char_models import HighwayNetwork, CharacterLevelCNN
from dataset import DatasetPreprosessed, PreprocessingPOS

class DTranNER(nn.Module):
    def __init__(self, device, tag_to_ix, character_vocabulary_size, word_embedding_dim, hidden_dim, pp_hidden_dim, dropout, 
                 filter_num_width, dropout_ratio=0.5, pp_bilinear=False, pp_bilinear_pooling=False, bilinear_dim=300, monitor=None):
        super(DTranNER, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_dim_pp = pp_hidden_dim
        self.dropout = dropout
        self.pp_bilinear = pp_bilinear
        self.pp_bilinear_pooling = pp_bilinear_pooling
        self.bilinear_dim = bilinear_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.monitor = monitor
        self.pos_embedding_dim = len(PreprocessingPOS.__TAGS__)
#         self.embedding_dim = word_embedding_dim + sum([n_filter for width, n_filter in filter_num_width.items()])
        self.embedding_dim = word_embedding_dim + self.pos_embedding_dim + sum([n_filter for width, n_filter in filter_num_width.items()]) 
        print(self.embedding_dim, filter_num_width)

        self.character_cnn = CharacterLevelCNN(device, character_vocabulary_size, filter_num_width, "unary", monitor).to(device=device)
        self.character_cnn_pp = CharacterLevelCNN(device, character_vocabulary_size, filter_num_width, "pairwise", monitor).to(device=device)
        
#         self.pos_embedding = nn.Embedding(self.pos_embedding_dim, self.pos_embedding_dim).to(device=device)
#         nn.init.orthogonal_(self.pos_embedding.weight, gain=2)

#         self.pos_embedding = nn.Parameter(torch.randn(self.pos_embedding_dim, self.pos_embedding_dim, requires_grad=False).to(device=device), requires_grad=False)
        self.pos_embedding = nn.Parameter(torch.eye(self.pos_embedding_dim, requires_grad=False).to(device=device), requires_grad=False)
#         self.pos_embedding = nn.Parameter(torch.rand(self.pos_embedding_dim, self.pos_embedding_dim), requires_grad=False).to(device=device)
#         nn.init.orthogonal_(self.pos_embedding.data, gain=2)
        
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=1, bidirectional=True).to(device=device)
        self.lstm_pp = nn.LSTM(self.embedding_dim, self.hidden_dim_pp, num_layers=1, bidirectional=True).to(device=device)
                
        if self.dropout:
            self.dropout_embedding = nn.Dropout(p=dropout_ratio).to(device=device)
            self.dropout_embedding_pp = nn.Dropout(p=dropout_ratio).to(device=device)
#             self.dropout_u = nn.Dropout(p=dropout_ratio).to(device=device)
#             self.dropout_pp = nn.Dropout(p=dropout_ratio).to(device=device)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = weight_norm(nn.Linear(2*hidden_dim, self.tagset_size).to(device=device), dim=None)
        
        if self.pp_bilinear_pooling:
            self.U = weight_norm(nn.Linear(2*self.hidden_dim_pp, 2*self.hidden_dim_pp, bias=False).to(device=device), dim=None)
#             nn.init.orthogonal_(self.U.weight, gain=2)
            self.V = weight_norm(nn.Linear(2*self.hidden_dim_pp, 2*self.hidden_dim_pp, bias=False).to(device=device), dim=None)
#             nn.init.orthogonal_(self.V.weight, gain=2)
            self.P = weight_norm(nn.Linear(2*self.hidden_dim_pp, self.bilinear_dim).to(device=device), dim=None)
#             nn.init.orthogonal_(self.P.weight, gain=2)
            self.hidden2tag_pp = weight_norm(nn.Linear(self.bilinear_dim, self.tagset_size**2).to(device=device), dim=None)
        elif self.pp_bilinear:
            self.bilinear = nn.Bilinear(2*self.hidden_dim_pp, 2*self.hidden_dim_pp, self.bilinear_dim, bias=False).to(device=device)
            self.hidden2tag_pp = nn.Linear(self.bilinear_dim, self.tagset_size**2).to(device=device)
        else:
            self.hidden2tag_pp = weight_norm(nn.Linear(6*self.hidden_dim_pp, self.tagset_size**2).to(device=device), dim=None)
        # Matrix of transition parameters.  Entry i,j is the score of # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size, requires_grad=True).to(device=device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[DatasetPreprosessed.__START_TAG__], :] = -100.
        self.transitions.data[:, tag_to_ix[DatasetPreprosessed.__STOP_TAG__]] = -100.

        self.hidden = self.init_hidden()
        self.hidden_pp = self.init_encoder_hidden()

    def init_hidden(self):
        hidden = torch.zeros((2, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True)
        cell = torch.zeros((2, 1, self.hidden_dim), dtype=torch.float, device=self.device, requires_grad=True) 
        return (hidden, cell)
#         return (torch.randn(2, 1, self.hidden_dim // 2, requires_grad=True).cuda(self.device), torch.randn(2, 1, self.hidden_dim // 2, requires_grad=True).cuda(self.device))
        
    def init_encoder_hidden(self):
        hidden = torch.zeros((2, 1, self.hidden_dim_pp), dtype=torch.float, device=self.device, requires_grad=True)
        cell = torch.zeros((2, 1, self.hidden_dim_pp), dtype=torch.float, device=self.device, requires_grad=True) 
        return (hidden, cell)
#         return (torch.randn(2, 1, self.hidden_dim // 2, requires_grad=True).cuda(self.device), torch.randn(2, 1, self.hidden_dim // 2, requires_grad=True).cuda(self.device))
    
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
#         init_alphas = torch.randn(1, self.tagset_size, dtype=torch.float, requires_grad=True).to(device=self.device)
        init_alphas = torch.full((1, self.tagset_size), -100, dtype=torch.float, requires_grad=True).to(device=self.device)
        # START_TAG has all of the score.
#         init_alphas = feats[0].view(1, -1).expand(1, self.tagset_size) + self.transitions[self.tag_to_ix[START_TAG]]
        init_alphas[0][self.tag_to_ix[DatasetPreprosessed.__START_TAG__]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).expand(1, self.tagset_size)
                assert emit_score.size() == trans_score.size()
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
#                 print(next_tag_var, next_tag_var.size())
#                 print(utils.log_sum_exp(next_tag_var).view(1))
                alphas_t.append(utils.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__]]
        alpha = utils.log_sum_exp(terminal_var)
        return alpha

    def _forward_alg_pp(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -100, dtype=torch.float, requires_grad=True).to(device=self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[DatasetPreprosessed.__START_TAG__]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats[:-1]:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = feat.view(self.tagset_size, self.tagset_size)[next_tag].view(1, -1).expand(1, self.tagset_size)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(utils.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + feats[-1].view(self.tagset_size, self.tagset_size)[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__]]
        alpha = utils.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, iter):
        self.hidden = self.init_hidden()
        embeds = sentence.view(-1, 1, self.embedding_dim)
        if self.dropout:
            embeds = self.dropout_embedding(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(-1, 2*self.hidden_dim)
#         if self.dropout:
#             lstm_out = self.dropout_u(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_pp_features(self, sentence, iter):
        self.hidden_pp = self.init_encoder_hidden()
        embeds = sentence.view(-1, 1, self.embedding_dim)

        if self.dropout:
            embeds = self.dropout_embedding_pp(embeds)
        
        lstm_out_pp, self.hidden_pp = self.lstm_pp(embeds, self.hidden_pp)

#         print(lstm_out_pp.size())
        if self.pp_bilinear_pooling:
            U = self.U(lstm_out_pp[:-1])
            V = self.V(lstm_out_pp[1:])
            h = torch.einsum('ijk,ijk->ijk', [U, V])
            pp = self.P(F.relu(h))
#             pp = F.relu(pp)
#             h = torch.tanh(torch.einsum('ijk,ijk->ijk', [U, V]))
#             pp = self.P(torch.tanh(torch.einsum('ijk,ijk->ijk', [U, V])))
#             torch.einsum('[self.U(lstm_out_pp[:-1], self.V(lstm_out_pp[1:])])
            
        elif self.pp_bilinear:
            atv = self.bilinear(lstm_out_pp[:-1], lstm_out_pp[1:])
            if self.monitor and iter:
                name = "activation in pairwise bilinear before tanh"
                self.monitor.add_histogram(name, atv, iter)
#             pp = torch.tanh(atv)
            pp = F.relu(atv)
        else:
#             pp = torch.cat([lstm_out_pp[:-1], lstm_out_pp[1:], torch.mul(lstm_out_pp[:-1], lstm_out_pp[1:])], 2)
            pp = torch.cat([lstm_out_pp[:-1], lstm_out_pp[1:], torch.einsum('ijk,ijk->ijk', [lstm_out_pp[:-1], lstm_out_pp[1:]])], 2)

#         if self.dropout:
#             pp = self.dropout_pp(pp)

        pp_feats = self.hidden2tag_pp(pp)
            
        return pp_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.tensor(0., dtype=torch.float, requires_grad=True).to(device=self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[DatasetPreprosessed.__START_TAG__]], dtype=torch.int).to(device=self.device), tags.squeeze()])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__], tags[-1]]
        return score

    def _score_sentence_pp(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.tensor(0., dtype=torch.float, requires_grad=True).to(device=self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[DatasetPreprosessed.__START_TAG__]], dtype=torch.int).to(device=self.device), tags.squeeze()])
        for i, feat in enumerate(feats[:-1]):
            score = score + feat.view(self.tagset_size, self.tagset_size)[tags[i + 1], tags[i]]
        score = score + feats[-1].view(self.tagset_size, self.tagset_size)[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__], tags[-1]]
        return score

    def _viterbi_decode(self, feats, feats_pp):
        backpointers = []
 
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -100., dtype=torch.float).to(device=self.device)
        init_vvars[0][self.tag_to_ix[DatasetPreprosessed.__START_TAG__]] = 0
        
#         init_vvars = feats[0].view(1, -1).expand(1, self.tagset_size) + self.transitions[self.tag_to_ix[START_TAG]]
 
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat, feat_pp in zip(feats, feats_pp[:-1]):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
 
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag] + feat_pp.view(self.tagset_size, self.tagset_size)[next_tag].view(1, -1).expand(1, self.tagset_size)
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
 
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__]] + feats_pp[-1].view(self.tagset_size, self.tagset_size)[self.tag_to_ix[DatasetPreprosessed.__STOP_TAG__]]
        best_tag_id = utils.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
 
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
#         assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, pos, characters, tags, iter):
        char_feats = self.character_cnn(characters, iter)
#         pos_embedded = self.pos_embedding(pos)
        pos_embedded = self.pos_embedding[pos]
#         print(sentence.size(), pos_embedded.size(), char_feats.size(), pos.size())
#         if sentence.size(1) != pos.size(1):
#             print(pos)
        feats = self._get_lstm_features(torch.cat((sentence, pos_embedded, torch.unsqueeze(char_feats, 0)), 2), iter)
#         feats = self._get_lstm_features(torch.cat((sentence, torch.unsqueeze(char_feats, 0)), 2), iter)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def neg_log_likelihood_pp(self, sentence, pos, characters, tags, iter):
        char_feats = self.character_cnn_pp(characters, iter)
        char_feats = torch.cat((torch.tensor(np.zeros((1, char_feats.size(1)), dtype=np.float32), dtype=torch.float, requires_grad=True).to(device=self.device),
                                char_feats,
                                torch.tensor(np.zeros((1, char_feats.size(1)), dtype=np.float32), dtype=torch.float, requires_grad=True).to(device=self.device)), 0)
#         pos_embedded = self.pos_embedding(pos)
        pos_embedded = self.pos_embedding[pos]
#         feats = self._get_lstm_pp_features(torch.cat((sentence, torch.unsqueeze(char_feats, 0)), 2), iter)
        feats = self._get_lstm_pp_features(torch.cat((sentence, pos_embedded, torch.unsqueeze(char_feats, 0)), 2), iter)
        forward_score = self._forward_alg_pp(feats)
        gold_score = self._score_sentence_pp(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, sentence_pp, pos, pos_pp, characters):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        char_feats = self.character_cnn(characters)
        char_feats_pp = self.character_cnn_pp(characters)
#         pos_embedded = self.pos_embedding(pos)
#         pos_embedded_pp = self.pos_embedding(pos_pp)
        pos_embedded = self.pos_embedding[pos]
        pos_embedded_pp = self.pos_embedding[pos_pp]

        lstm_feats = self._get_lstm_features(torch.cat((sentence, pos_embedded, torch.unsqueeze(char_feats, 0)), 2), None)
#         lstm_feats = self._get_lstm_features(torch.cat((sentence, torch.unsqueeze(char_feats, 0)), 2), None)
        char_feats_pp = torch.cat((torch.tensor(np.zeros((1, char_feats_pp.size(1)), dtype=np.float32), dtype=torch.float).to(device=self.device),
                                   char_feats_pp,
                                   torch.tensor(np.zeros((1, char_feats_pp.size(1)), dtype=np.float32), dtype=torch.float).to(device=self.device)), 0)
#         lstm_pp_feats = self._get_lstm_pp_features(torch.cat((sentence_pp, torch.unsqueeze(char_feats_pp, 0)), 2), None)
        lstm_pp_feats = self._get_lstm_pp_features(torch.cat((sentence_pp, pos_embedded_pp, torch.unsqueeze(char_feats_pp, 0)), 2), None)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, lstm_pp_feats)
        return score, tag_seq

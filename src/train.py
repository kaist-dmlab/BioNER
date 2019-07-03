import os
import sys
import argparse
import csv
import math
import time
import pickle
import numpy as np
import gensim
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.DTranNER import DTranNER
from models.BiLSTM_CRF import BiLSTM_CRF
import Evaluation as conlleval
from dataset import DatasetPreprosessed, PreprocessingPOS
from tensorboardX import SummaryWriter

__DEFAULT_CNN_FILTER_NUM_WIDTH__ = {1:60, 2:200, 3:200, 4:200, 5:200, 6:200}

def setup_parser():
    parser = argparse.ArgumentParser('DTranNER with Deep Learning-based Label-Label Transition Model ')
    parser.add_argument('--disable_cuda', action='store_true', default=False)
    parser.add_argument('--minibatch_size', type=int, default='10')
    parser.add_argument('--tagging_scheme', type=str, default='bioes')
    parser.add_argument('--dataset_name', type=str, default='BC5CDR')
    parser.add_argument('--epoch', type=int, default='100')
    parser.add_argument('--learning_rate', type=float, default='0.001')
    parser.add_argument('--clip_grad', type=float, default='5.')
    parser.add_argument('--character_token_valid_max_length', type=int, default='50')
    parser.add_argument('--word_embedding_dim', type=int, default='200')
    parser.add_argument('--hidden_dim', type=int, default='800')
    parser.add_argument('--pp_hidden_dim', type=int, default='500')
    parser.add_argument('--dropout', action='store_true', default=False)
    parser.add_argument('--dropout_ratio', type=float, default='0.25')
    parser.add_argument('--pp_bilinear', action='store_true', default=False)
    parser.add_argument('--pp_bilinear_pooling', action='store_true', default=False)
    parser.add_argument('--bilinear_dim', type=int, default='300')
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('-d', '--DTranNER', action='store_true', default=False)
    parser.add_argument('--debugging', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--torch_rand_seed', type=int, default='4123')
    
    return parser

def main(args):
    print(args)
    train_dataset = DatasetPreprosessed(args.tagging_scheme, args.dataset_name, 
                                        args.DTranNER, 
                                        args.word_embedding_dim, 
                                        args.character_token_valid_max_length, 
                                        train=True)
    test_dataset = DatasetPreprosessed(args.tagging_scheme, args.dataset_name,
                                       args.DTranNER,
                                       args.word_embedding_dim,
                                       args.character_token_valid_max_length,                                                
                                       train=False)
        
    print('Dataset Created.')

    if args.gpu >= 0 and torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.torch_rand_seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.torch_rand_seed)

    torch.set_default_tensor_type(torch.FloatTensor)
    
    print("Device:%s" %device)
    
    train_params = {"batch_size": 1,
                    "shuffle": True,
                    "num_workers": 1}
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "num_workers": 1}

    train_generator = DataLoader(train_dataset, **train_params)
    test_generator = DataLoader(test_dataset, **test_params)

    train_dataset.label_dict[DatasetPreprosessed.__START_TAG__] = len(train_dataset.label_dict)
    train_dataset.label_dict[DatasetPreprosessed.__STOP_TAG__] = len(train_dataset.label_dict)

    writer = SummaryWriter() if args.tensorboard else None 

    if args.DTranNER:
        model = DTranNER(device, train_dataset.label_dict, 
                                DatasetPreprosessed.character_vocabulary_size(), 
                                args.word_embedding_dim, args.hidden_dim, args.pp_hidden_dim, args.dropout, __DEFAULT_CNN_FILTER_NUM_WIDTH__, 
                                args.dropout_ratio, args.pp_bilinear, args.pp_bilinear_pooling, args.bilinear_dim, writer)
    else:
        model = BiLSTM_CRF(train_dataset.label_dict, args.word_embedding_dim, args.hidden_dim, args.dropout)

    if torch.cuda.is_available() and not args.disable_cuda:
        model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    num_iter_per_epoch = len(train_generator)
    print("# of iteration each epoch = %d" %num_iter_per_epoch)
        
    log_file_name = 'DTranNER_'
    log_file_name += 'dataset_' + str(args.dataset_name)
    log_file_name += '_' + time.strftime('%H', time.localtime()) + '_hour'
    log_file_name += '.txt'

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())
#     with open(log_file_name, 'w', 0) as f_log:
#         f_log.write('Training Start...\n')
#     writer.add_graph(model, verbose=False)
    
    n_minibatch = args.minibatch_size
    best_performance = {'epoch':0, 'fscore':0., 'precision':0., 'recall':0.}
    loss = 0.
    if args.DTranNER:
        loss_unary, loss_pairwise = 0., 0.

    for name, param in model.named_parameters():
        if name == 'transitions':
            print(param.clone().cpu().data.numpy())

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        for iter, batch in enumerate(train_generator):
            if args.DTranNER:
                feature, feature_pp, pos, pos_pp, characters, label = batch
            else:
                feature, characters, label = batch
            if torch.cuda.is_available():
                feature = feature.to(device=device)
                characters = characters.to(device=device)
                label = label.to(device=device)
                pos = pos.to(device=device)
                if args.DTranNER:
                    feature_pp = feature_pp.to(device=device)
                    pos_pp = pos_pp.to(device=device)

            if args.DTranNER:
                each_loss_unary = model.neg_log_likelihood(feature, pos, characters, label, epoch*num_iter_per_epoch+iter)
                each_loss_pairwise = model.neg_log_likelihood_pp(feature_pp, pos_pp, characters, label, epoch*num_iter_per_epoch+iter)
                loss_unary += each_loss_unary 
                loss_pairwise += each_loss_pairwise
                loss += each_loss_unary + each_loss_pairwise
            else:
                loss += model.neg_log_likelihood(feature, characters, label)
    
            if (iter+1) % n_minibatch == 0 or (iter+1) % num_iter_per_epoch == 0:
#                 if (iter+1) % n_minibatch == 0 or (iter+1) % num_iter_per_epoch == 0:
                if args.DTranNER:
                    print("In %d epoch, %d: loss=%.5f loss_unary=%.5f loss_pairwise=%.5f" %(epoch, iter, loss.item(), loss_unary.item(), loss_pairwise.item()))
                else:
                    print("In %d epoch, %d: loss=%.5f" %(epoch, iter, loss.item()))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                loss = 0.
                if args.DTranNER:
                    loss_unary, loss_pairwise = 0., 0.
                optimizer.zero_grad()
            
#             if args.tensorboard:
#                 for name, param in model.named_parameters():
#                     writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)
        
#         for name, param in model.named_parameters():
#             if name == 'transitions':
#                 print(param.clone().cpu().data.numpy())
        print("Faulty Sentences for POS in train = %d" %PreprocessingPOS.n_fault_pos_tagging_sentences)
        PreprocessingPOS.n_fault_pos_tagging_sentences = 0
        
        model.eval()
        with torch.no_grad():
            evaluate_list = list()
            for iter, batch in enumerate(test_generator):
                if args.DTranNER:
                    sentence, feature, feature_pp, pos, pos_pp, characters, label = batch
                    if torch.cuda.is_available():
                        feature = feature.to(device=device)
                        feature_pp = feature_pp.to(device=device)
                        pos = pos.to(device=device)
                        pos_pp = pos_pp.to(device=device)
                        characters = characters.to(device=device)
                    _, prediction = model(feature, feature_pp, pos, pos_pp, characters)
                else:
                    sentence, feature, characters, label = batch
                    if torch.cuda.is_available():
                        feature = feature.to(device=device)
                        characters = characters.to(device=device)
                    _, prediction = model(feature, characters)
                for word, target, predict in zip(sentence, label.squeeze(), prediction):
                    evaluate_str = '{} {} {}'.format(word, train_dataset.label_dict_reversed[target.item()], train_dataset.label_dict_reversed[predict])
                    evaluate_list.append(evaluate_str)
            counts = conlleval.evaluate(evaluate_list)
            precision, recall, fscore = conlleval.report(counts)
            if fscore > best_performance['fscore']:
                best_performance = {'epoch':epoch, 'fscore':fscore, 'precision':precision, 'recall':recall}
            print("Best Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(best_performance['epoch'], 
                                                                                             best_performance['precision'], 
                                                                                             best_performance['recall'], 
                                                                                             best_performance['fscore']))

        print("Faulty Sentences for POS in test = %d" %PreprocessingPOS.n_fault_pos_tagging_sentences)
        PreprocessingPOS.n_fault_pos_tagging_sentences = 0

    if args.tensorboard:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()   

if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()

    main(args)

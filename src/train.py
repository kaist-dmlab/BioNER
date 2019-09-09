import os
import sys
import argparse
import csv
import math
import time
import pickle
import numpy as np
import collections
import logging
import logging.handlers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.CRF_lstm import CRF_FB
from models.CRF_baseline import CRF_baseline
from models.CRF_united import CRF_united
import Evaluation as conlleval
from dataset import DatasetPreprosessed
from tensorboardX import SummaryWriter

__DEFAULT_CNN_FILTER_NUM_WIDTH__ = {2:200, 3:200}
__CHAR_MAX_LENGTH__ = 50

def setup_parser():
    parser = argparse.ArgumentParser('DTranNER with Deep Learning-based Label-Label Transition Model ')
    # Required parameters
    
    # Other parameters
    parser.add_argument('--disable_cuda', action='store_true', default=False)
    parser.add_argument('--minibatch_size', type=int, default='10')
    parser.add_argument('--tagging_scheme', type=str, default='bioes')
    parser.add_argument('--dataset_name', type=str, default='BC5CDR')
    parser.add_argument('--attn', type=str, default='multi')
    parser.add_argument('--epoch', type=int, default='100')
    parser.add_argument('--learning_rate', type=float, default='0.001')
    parser.add_argument('--clip_grad', type=float, default='5.')
    parser.add_argument('--elmo_embedding_dim', type=int, default='1224')
    parser.add_argument('--elmo_n_layers', type=int, default='1')
    parser.add_argument('--elmo_dropout_rate', type=float, default='0.')
    parser.add_argument('--hidden_dim', type=int, default='800')
    parser.add_argument('--pp_hidden_dim', type=int, default='500')
    parser.add_argument('--dropout_ratio', type=float, default='0.25')
    parser.add_argument('--pp_bilinear', action='store_true', default=False)
    parser.add_argument('--pp_bilinear_pooling', action='store_true', default=False)
    parser.add_argument('--bilinear_dim', type=int, default='300')
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--DTranNER', action='store_true', default=False)
    parser.add_argument('--debugging', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default='4123')
    parser.add_argument('--crf_fb', action='store_true', default=False)
    parser.add_argument('--crf_united', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--gate_bias', type=float, default='-1.')
    parser.add_argument('--weight_decay', type=float, default='1e-5')
    parser.add_argument('--char_cnn', action='store_true', default=False)
    parser.add_argument('--pairwise_gate', action='store_true', default=False)
    parser.add_argument('--working_test', action='store_true', default=False)
    parser.add_argument('--pairwise_query_type', type=str, default='mul')
    parser.add_argument('--normalization_type', type=str, default=None)
    parser.add_argument('--train_type', type=str, default='sequence') # or fb
    parser.add_argument('--n_train_iter', type=int, default='1000')
    parser.add_argument('--n_test_iter', type=int, default='500')
    parser.add_argument('--all_test', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--loading', action='store_true', default=False)
    parser.add_argument('--optim', type=str, default="adam")
    parser.add_argument('--shared_lstm', action='store_true', default=False)
    parser.add_argument('--inp_config', type=str, default="full")
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--n_layers', type=int, default='1')
    
    return parser

def setup_logger(args):
    logger = logging.getLogger('DTranNER')
    logger.setLevel(logging.INFO)

    log_file_name = 'DTranNER_'
    log_file_name += 'dataset_' + str(args.dataset_name) + '_' + str(args.hidden_dim) + '_' + str(args.pp_hidden_dim) + '_' + str(args.bilinear_dim)
    log_file_name += '.txt'

    fileHandler = logging.FileHandler(log_file_name, mode='w')
    streamHandler = logging.StreamHandler()

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    return logger

def validation(model, test_generator, test_dataset, device):
    print("Running Evaluation with pre-trained Model")
    model.eval()
    with torch.no_grad():
        evaluate_list = list()
        for iter, batch in enumerate(test_generator):
            sentence, elmo, word_idxs, elmo_pp, label = batch
            if torch.cuda.is_available():
                elmo = elmo.to(device=device)
                word_idxs = word_idxs.to(device=device)
                elmo_pp = elmo_pp.to(device=device)
                _, prediction = model(elmo, word_idxs, elmo_pp)
            try:
                for word, target, predict in zip(sentence, label.squeeze(), prediction):
                    evaluate_str = '{} {} {}'.format(word, test_dataset.label_dict_reversed[target.item()], test_dataset.label_dict_reversed[predict])
                    evaluate_list.append(evaluate_str)
            except Exception as ex:
                print("Error = ", ex)
                pass
            
        counts = conlleval.evaluate(evaluate_list)
        precision, recall, fscore = conlleval.report(counts)
        print("Performance: Precision = %.4f Recall = %.4f F-Score = %.4f" %(precision, recall, fscore))
    
def main(args):
    
    logger = setup_logger(args)        
    logger.info(args)

    if args.gpu >= 0 and torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)

    torch.set_default_tensor_type(torch.FloatTensor)
    
    logger.info("Device:%s" %device)

    train_dataset = DatasetPreprosessed(args.tagging_scheme, args.dataset_name, 
                                        args.elmo_embedding_dim, 
                                        args.elmo_dropout_rate,
                                        args.elmo_n_layers, 
                                        __CHAR_MAX_LENGTH__,
                                        True if args.inp_config == "full" or args.inp_config == "w2v" else False, 
                                        args.char_cnn,
                                        device=device,
                                        train=True)
    test_dataset = DatasetPreprosessed(args.tagging_scheme, args.dataset_name,
                                       args.elmo_embedding_dim,
                                       args.elmo_dropout_rate,
                                       args.elmo_n_layers, 
                                       __CHAR_MAX_LENGTH__,
                                       True if args.inp_config == "full" or args.inp_config == "w2v" else False, 
                                       args.char_cnn,
                                       device=device,
                                       train=False)
        
    logger.info('Dataset Created.')
    
    train_params = {"batch_size": 1,
                    "shuffle": True,
                    "pin_memory": True,
                    "num_workers": 3}
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "pin_memory": True,
                   "num_workers": 3}

    train_generator = DataLoader(train_dataset, **train_params)
    test_generator = DataLoader(test_dataset, **test_params)

    train_dataset.label_dict[DatasetPreprosessed.__START_TAG__] = len(train_dataset.label_dict)
    train_dataset.label_dict_reversed[len(train_dataset.label_dict)-1] = DatasetPreprosessed.__START_TAG__
    train_dataset.label_dict[DatasetPreprosessed.__STOP_TAG__] = len(train_dataset.label_dict)
    train_dataset.label_dict_reversed[len(train_dataset.label_dict)-1] = DatasetPreprosessed.__STOP_TAG__
    print("Label dictionary = ", train_dataset.label_dict)
    print("Label dictionary indexes = ", train_dataset.label_dict_reversed)

    writer = SummaryWriter() if args.tensorboard else None 

    if args.baseline:
        model = CRF_baseline(device, train_dataset.label_dict, args.hidden_dim, args.dropout_ratio, args.bilinear_dim, args.elmo_embedding_dim, args.attn, writer)
    elif args.crf_united:
        model = CRF_united(device, train_dataset.label_dict, args.hidden_dim, args.dropout_ratio, args.bilinear_dim, args.elmo_embedding_dim, args.attn, writer)
    elif args.crf_fb:
        n_chars = len(DatasetPreprosessed.CHARACTER_VOCABULARY)
        model = CRF_FB(device, train_dataset.label_dict, args.n_layers, args.hidden_dim, args.pp_hidden_dim, args.char_cnn, 
                       n_chars, __DEFAULT_CNN_FILTER_NUM_WIDTH__, args.pairwise_gate, args.train_type, args.normalization_type, 
                       args.elmo_dropout_rate, args.dropout_ratio, args.shared_lstm, args.inp_config,
                       args.pairwise_query_type, args.bilinear_dim, args.elmo_embedding_dim, args.attn, args.all_test,
                       args.gate_bias, writer, logger)
    else:
        model = CRF(device, train_dataset.label_dict, args.hidden_dim, args.dropout_ratio, args.bilinear_dim, args.elmo_embedding_dim, args.attn, writer)

    if torch.cuda.is_available() and not args.disable_cuda:
        model.to(device=device)

    if args.optim.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    num_iter_per_epoch = len(train_generator)
    logger.info("# of iteration each epoch = %d" %num_iter_per_epoch)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())

    n_minibatch = args.minibatch_size
    best_performance = {'epoch':0, 'fscore':0., 'precision':0., 'recall':0.}
    if args.all_test:
        best_performance_unary = {'epoch':0, 'fscore':0., 'precision':0., 'recall':0.}
        best_performance_pairwise = {'epoch':0, 'fscore':0., 'precision':0., 'recall':0.}
    loss, loss_unary, loss_pairwise = 0., 0., 0.

    np.set_printoptions(precision=4)

    def print_parameter(name):
        print(train_dataset.label_dict)
        for param_name, param in model.named_parameters():
            if param_name == name:
                print(param.clone().cpu().data.numpy())

    n_epoch = 1 if args.working_test else args.epoch  
    
    MODEL_PATH = "{0}_{1}_{2}_{3}_{4}.pt".format(args.dataset_name, str(args.hidden_dim), str(args.pp_hidden_dim), str(args.bilinear_dim), args.train_type)
    
    if args.loading:
        print("Loading")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        validation(model, test_generator, test_dataset, device)
        sys.exit()
    
    for epoch in range(n_epoch):
        logger.info("Running Model Train")
        model.train()
        optimizer.zero_grad()
        for iter, batch in enumerate(train_generator):
            elmo_feature, word_idxs, chars, label = batch
            if torch.cuda.is_available():
                elmo_feature = elmo_feature.to(device=device)
                word_idxs = word_idxs.to(device=device)
                chars = chars.to(device=device)
                label = label.to(device=device)

            if args.baseline:
                each_loss, each_loss_unary = model.neg_log_likelihood(elmo_feature, label, epoch*num_iter_per_epoch+iter)
                loss_unary += each_loss_unary
                loss += each_loss
            elif args.crf_united:
                each_loss, each_loss_unary, each_loss_pairwise = model.neg_log_likelihood(elmo_feature, elmo_feature_pp, label, epoch*num_iter_per_epoch+iter)
                loss_unary += each_loss_unary
                loss_pairwise += each_loss_pairwise
                loss += each_loss
            elif args.crf_fb:
                each_loss, each_loss_unary, each_loss_pairwise = model.neg_log_likelihood(elmo_feature, word_idxs, chars, label, epoch*num_iter_per_epoch+iter)
                loss_unary += each_loss_unary
                loss_pairwise += each_loss_pairwise
                loss += each_loss
            else:
                each_loss_unary = model.neg_log_likelihood(elmo_feature, label, epoch*num_iter_per_epoch+iter)
                each_loss_pairwise = model.neg_log_likelihood_pp(elmo_feature_pp, label, epoch*num_iter_per_epoch+iter)
                loss_unary += each_loss_unary 
                loss_pairwise += each_loss_pairwise
                loss += each_loss_unary + each_loss_pairwise
    
            if (iter+1) % n_minibatch == 0 or (iter+1) % num_iter_per_epoch == 0:
                if args.baseline:
                    logger.info("In %d epoch, %d: loss=%.5f loss_unary=%.5f" %(epoch, iter, loss, loss_unary.item()))
                elif args.crf_fb or args.crf_united:
                    logger.info("In %d epoch, %d: loss=%.5f loss_unary=%.5f loss_pairwise=%.5f" %(epoch, iter, loss, loss_unary.item(), loss_pairwise.item()))
                else:
                    logger.info("In %d epoch, %d: loss=%.5f loss_unary=%.5f loss_pairwise=%.5f" %(epoch, iter, loss.item(), loss_unary.item(), loss_pairwise.item()))

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        print("No gradient in ", name)

                optimizer.step()
                loss, loss_unary, loss_pairwise = 0., 0., 0.
                optimizer.zero_grad()
                            
            if (iter+1) % num_iter_per_epoch == 0:
                logger.info("Running Evaluation")
                model.eval()
                if args.logging:
                    output_file = "./output_{0}_{1}_{2}_{3}_{4}.txt".format(args.dataset_name, str(args.hidden_dim), str(args.pp_hidden_dim), str(args.bilinear_dim), str(epoch))
                    eval_f = open(output_file, 'w')
                with torch.no_grad():
                    evaluate_list = list()
                    if args.all_test:
                        evaluate_list_unary = list()
                        evaluate_list_pairwise = list()
                    for iter, batch in enumerate(test_generator):
                        sentence, elmo, word_idxs, elmo_pp, label = batch
                        if torch.cuda.is_available():
                            elmo = elmo.to(device=device)
                            word_idxs = word_idxs.to(device=device)
                            elmo_pp = elmo_pp.to(device=device)
                            if args.all_test:
                                score, prediction, score_u, prediction_unary, score_p, prediction_pairwise = model(elmo, word_idxs, elmo_pp)
                                print(score, score_u, score_p)
                            else:
                                score, prediction = model(elmo, word_idxs, elmo_pp)
                        try:
                            for word, target, predict in zip(sentence, label.squeeze(), prediction):
                                evaluate_str = '{} {} {}'.format(word, train_dataset.label_dict_reversed[target.item()], train_dataset.label_dict_reversed[predict])
                                evaluate_list.append(evaluate_str)

                            if args.logging:
                                evaluate_str = evaluate_str + "\n" 
                                eval_f.write(evaluate_str)
                            if args.all_test:
                                for word, target, predict in zip(sentence, label.squeeze(), prediction_unary):
                                    evaluate_str = '{} {} {}'.format(word, train_dataset.label_dict_reversed[target.item()], train_dataset.label_dict_reversed[predict])
                                    evaluate_list_unary.append(evaluate_str)
                                for word, target, predict in zip(sentence, label.squeeze(), prediction_pairwise):
                                    evaluate_str = '{} {} {}'.format(word, train_dataset.label_dict_reversed[target.item()], train_dataset.label_dict_reversed[predict])
                                    evaluate_list_pairwise.append(evaluate_str)
                        except Exception as ex:
                            print("Error = ", ex)
                            print("sentence = ", sentence)
                            print("label = ", label)
                            print("prediction = ", prediction)
                            pass
                        if args.logging:
                            eval_f.write("\n")
                    if args.logging:
                        eval_f.close()
                
                    counts = conlleval.evaluate(evaluate_list)
                    precision, recall, fscore = conlleval.report(counts)
                    logger.info("Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(epoch, 
                                                                                                      precision,
                                                                                                      recall,
                                                                                                      fscore))

                    if fscore > best_performance['fscore']:
                        best_performance = {'epoch':epoch, 'fscore':fscore, 'precision':precision, 'recall':recall}
                        if args.save:
                            logger.info("Model saved {}".format(MODEL_PATH))
                            torch.save(model.state_dict(), MODEL_PATH)

                    logger.info("Device:%s" %device)
                    logger.info("Best Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(best_performance['epoch'], 
                                                                                                         best_performance['precision'], 
                                                                                                         best_performance['recall'], 
                                                                                                         best_performance['fscore']))
                    if args.all_test:
                        counts = conlleval.evaluate(evaluate_list_unary)
                        precision, recall, fscore = conlleval.report(counts)
                        logger.info("Unary Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(epoch, 
                                                                                                          precision,
                                                                                                          recall,
                                                                                                          fscore))
                        if fscore > best_performance_unary['fscore']:
                            best_performance_unary = {'epoch':epoch, 'fscore':fscore, 'precision':precision, 'recall':recall}

                        logger.info("Unary Best Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(best_performance_unary['epoch'], 
                                                                                                             best_performance_unary['precision'], 
                                                                                                             best_performance_unary['recall'], 
                                                                                                             best_performance_unary['fscore']))

                        counts = conlleval.evaluate(evaluate_list_pairwise)
                        precision, recall, fscore = conlleval.report(counts)
                        logger.info("Pairwise Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(epoch, 
                                                                                                          precision,
                                                                                                          recall,
                                                                                                          fscore))
                        if fscore > best_performance_pairwise['fscore']:
                            best_performance_pairwise = {'epoch':epoch, 'fscore':fscore, 'precision':precision, 'recall':recall}

                        logger.info("Best Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(best_performance_pairwise['epoch'], 
                                                                                                             best_performance_pairwise['precision'], 
                                                                                                             best_performance_pairwise['recall'], 
                                                                                                             best_performance_pairwise['fscore']))
                model.train()
                
            if args.working_test and iter > 200:
                logger.info("Running Evaluation in Working Test")
                model.eval()
                with torch.no_grad():
                    evaluate_list = list()
                    for iter, batch in enumerate(test_generator):
                        sentence, elmo, word_idxs, elmo_pp, label = batch
                        if torch.cuda.is_available():
                            elmo = elmo.to(device=device)
                            word_idxs = word_idxs.to(device=device)
                            elmo_pp = elmo_pp.to(device=device)
                        _, prediction = model(elmo, word_idxs, elmo_pp)
        
                        for word, target, predict in zip(sentence, label.squeeze(), prediction):
                            evaluate_str = '{} {} {}'.format(word, train_dataset.label_dict_reversed[target.item()], train_dataset.label_dict_reversed[predict])
                            evaluate_list.append(evaluate_str)

                        if args.working_test and iter > 200:
                            break
        
                    counts = conlleval.evaluate(evaluate_list)
                    precision, recall, fscore = conlleval.report(counts)
                    if fscore > best_performance['fscore']:
                        best_performance = {'epoch':epoch, 'fscore':fscore, 'precision':precision, 'recall':recall}
                    logger.info("Best Performance: In %d, Precision = %.4f Recall = %.4f F-Score = %.4f" %(best_performance['epoch'], 
                                                                                                     best_performance['precision'], 
                                                                                                     best_performance['recall'], 
                                                                                                     best_performance['fscore']))
                break

    if args.tensorboard:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()   

if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()

    main(args)

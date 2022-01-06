import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from narm_torch import metric
from narm_torch.utils import collate_fn
from narm_torch.narm import NARM
from narm_torch.dataset import load_data, RecSysDataset

import os
import argparse
import pickle
import time
from srgnn_torch.pytorch_code.utils import build_graph, Data, split_validation
from srgnn_torch.pytorch_code.model import *
from collections import defaultdict
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    

    
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in enumerate(train_loader):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 
        # wandb.log({"epoch" : epoch, "loss" : loss})

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        # if i % log_aggr == 0:
        #     print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)' % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1), len(seq) / (time.time() - start)))

        start = time.time()



def validate(valid_loader, model, args):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in valid_loader:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = metric.evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr



## NARM 돌리기

def narm(train_fns, experiment, y_or_d, frac):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='exps/', help='dataset directory path: exps/diginetica/yoochoose1_4/yoochoose1_64')
    parser.add_argument('--test_file', default='exps/', help='dataset directory path: exps/diginetica/yoochoose1_4/yoochoose1_64')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
    parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
    parser.add_argument('--valid', action='store_true', help='test')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    args = parser.parse_args([])
    print(args)
    
    
    test_fn = f'{y_or_d[0]}{int(1/frac):03}_test.txt'
    
    for i, train_fn in enumerate(train_fns):
        
        
        
        args.train_file = f'exps/experiment{experiment}/{y_or_d}/{train_fn}'
        args.test_file = f'exps/experiment{experiment}/{y_or_d}/{test_fn}'
      
        train, test = load_data(args.train_file, args.test_file, valid_portion=args.valid_portion, valid=args.valid)
      
        train_data = RecSysDataset(train)
        test_data = RecSysDataset(test)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
        test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
        
        
        # item 개수 구하기 ##################################
        with open(f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_tra_seqs.pkl', 'rb') as q:
          tra_seqs_frac = pickle.load(q)
        uniqueitems = set()
        for s in tra_seqs_frac:
            uniqueitems |= set(s)
        print(f'아이템 개수 : {len(uniqueitems)}')
        print(f'{min(uniqueitems)} ~ {max(uniqueitems)}')
        n_items = len(uniqueitems)
        print(f'모델에 들어가는 수 : {n_items + 1}')
        ######################################################
      
        model = NARM(n_items + 1, args.hidden_size, args.embed_dim, args.batch_size).to(device)
      
        optimizer = optim.Adam(model.parameters(), args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
      
        hits = []
        mrrs = []
        for epoch in range(args.epoch):
            # train for one epoch
            scheduler.step(epoch = epoch)
            trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)
      
            recall, mrr = validate(test_loader, model, args)
            recall *= 100
            mrr *= 100
            hits.append(recall)
            mrrs.append(mrr)
      
            print(f'Epoch {epoch} : Recall@{args.topk}: {recall:.4f}, MRR@{args.topk}: {mrr:.4f} \t{train_fn} \t {i+1}/{len(train_fns)} \t {test_fn}')
        
        
        try:
            # 파일 하다 돌릴때마다 저장할것 (코랩 끊길수도있으니까)
            with open(f'exps/experiment{experiment}/result_narm_{y_or_d}/{y_or_d[0]}{int(1/frac):03}/{train_fn}_hits.pkl', 'wb') as q:
              pickle.dump(hits, q)
          
            with open(f'exps/experiment{experiment}/result_narm_{y_or_d}/{y_or_d[0]}{int(1/frac):03}/{train_fn}_mrrs.pkl', 'wb') as q:
              pickle.dump(mrrs, q)
              
            
        except:  # 만약 파일명 등의 문제로 저장이 안 될 경우 결과 리턴
            return hits, mrrs



# SRGNN 돌리기


def srgnn(train_fns, experiment, y_or_d, frac):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=30, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    
    opt = parser.parse_args()
    print(opt)
    
    test_fn = f'{y_or_d[0]}{int(1/frac):03}_test.txt'
    for i, train_fn in enumerate(train_fns):
        
        
        # item 개수 구하기 ##################################
        with open(f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_tra_seqs.pkl', 'rb') as q:
          tra_seqs_frac = pickle.load(q)
        uniqueitems = set()
        for s in tra_seqs_frac:
            uniqueitems |= set(s)
        print(f'아이템 개수 : {len(uniqueitems)}')
        print(f'{min(uniqueitems)} ~ {max(uniqueitems)}')
        n_items = len(uniqueitems)
        print(f'모델에 들어가는 수 : {n_items + 1}')
        ######################################################
    
    
        train_data = pickle.load(open(f'exps/experiment{experiment}/{y_or_d}/{train_fn}', 'rb'))
        test_data = pickle.load(open(f'exps/experiment{experiment}/{y_or_d}/{test_fn}', 'rb'))
    
        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)
    
        n_node = n_items + 1
        model = trans_to_cuda(SessionGraph(opt, n_node))
    
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0] 
        bad_counter = 0
    
        hits = []
        mrrs = []
    
        for epoch in range(opt.epoch):
            
            print('-------------------------------------------------------')
            print(f'train file : {train_fn} \t {i+1}/{len(train_fns)}')
            print(f'test file : {test_fn}')
            print('epoch: ', epoch)
            hit, mrr = train_test(model, train_data, test_data, epoch)
    
            hits.append(hit)
            mrrs.append(mrr)
    
            flag = 0
            if hit > best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
                flag = 1
            if mrr > best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
                flag = 1
            print(f'Best Result : \t Recall@20: {best_result[0]:.4f}, \tMMR@20: {best_result[1]:.4f}, Epoch:\t{best_epoch[0]},\t{best_epoch[1]}')
            print(f'Curr Result : \t Recall@20 : {hit:.4f},\tMMR@20: {mrr:.4f}')
            # bad_counter += 1 - flag
            # print(f'bad_counter : {bad_counter}')
            # if bad_counter >= opt.patience:
            #     break
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))
        print(f'{train_fn}____________________________' * 100)
    
        # 파일 하다 돌릴때마다 저장할것 (코랩 끊길수도있으니까)
        with open(f'exps/experiment{experiment}/result_srgnn_{y_or_d}/{y_or_d[0]}{int(1/frac)}/{train_fn}_hits.pkl', 'wb') as q:
            pickle.dump(hits, q)
    
        with open(f'exps/experiment{experiment}/result_srgnn_{y_or_d}/{y_or_d[0]}{int(1/frac)}/{train_fn}_mrrs.pkl', 'wb') as q:
            pickle.dump(mrrs, q)







































    
    
    
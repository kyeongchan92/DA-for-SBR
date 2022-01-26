# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:33:16 2022

@author: Admin
"""

import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np



def crop_seqs(iseqs):
    out_seqs = []
    labs = []
    for seq in iseqs:
        for i in range(1, len(seq)):
            tar = seq[-i]  # tar : 맨 뒤 아이템 하나씩 잘라 라벨로
            labs += [tar]
            out_seqs += [seq[:-i]]
    return out_seqs, labs



def generate_label(y_or_d, fn, dstgsh_idx):
  # print(fn, dstgsh_idx)

  simmet = fn[dstgsh_idx:dstgsh_idx+3]
  return simmet
  if y_or_d == 'yoochoose':

    if fn[dstgsh_idx] == 'coo':
      label = f'{fn[dstgsh_idx+1:dstgsh_idx+4]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'p':
      label = f'{fn[dstgsh_idx+1:dstgsh_idx+4]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'w':
      label = f'{fn[dstgsh_idx:dstgsh_idx+3]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'j':  # jaccard_apr, jaccard_win
      label = f'{fn[dstgsh_idx:dstgsh_idx+11]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    else:
      if fn[dstgsh_idx+1] == 'r':
        label = f'yoochoose original'
      elif fn[dstgsh_idx+1] == 'a':  # tanimoto
        label = f'{fn[dstgsh_idx:13]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
      else:
        label = fn

  elif y_or_d == 'diginetica':
    if fn[dstgsh_idx] == 'c':
      if fn[dstgsh_idx+2] == 's':
        label = f'{fn[dstgsh_idx:+10]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
      else:
        label = f'{fn[dstgsh_idx:5]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'p':
      label = f'{fn[dstgsh_idx+1:7]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'w':
      label = f'{fn[dstgsh_idx:dstgsh_idx+3]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    elif fn[dstgsh_idx] == 'j':  # jaccard_apr, jaccard_win
      label = f'{fn[dstgsh_idx:dstgsh_idx+11]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
    else:  # train, tanimoto
      if fn[dstgsh_idx+1] == 'r':
        label = f'diginetica original'
      else:  # tanimoto
        label = f'{fn[dstgsh_idx:dstgsh_idx+8]}, l : {fn[-26]}, ps : {fn[-22:-19]}, pl : {fn[-16:-13]}'
  else:
    print('y_or_d error')
    return None
  
  return label


def cls_sorting(fn, cls_idx):
  if fn[cls_idx:cls_idx+2] == 'HH':
    if fn[cls_idx:cls_idx+4] == 'HHLH':
      return 5
    elif fn[cls_idx:cls_idx+8] == 'HHHLLHLL':
      return 6
    else:
      return 1
  elif fn[cls_idx:cls_idx+2] == 'LH':
    return 2
  elif fn[cls_idx:cls_idx+2] == 'HL':
    return 3
  elif fn[cls_idx:cls_idx+2] == 'LL':
    return 4
  else:
    return 0




def sort_fn(fn, dstgsh_idx, cls_idx):
  dstgsh_met = fn[dstgsh_idx:dstgsh_idx+3]
  cls_digit = cls_sorting(fn, cls_idx)
  if dstgsh_met == 'tra': 
    return 10 + cls_digit
  elif dstgsh_met == 'con':
    return 20 + cls_digit
  elif dstgsh_met == 'pmi':
    return 30 + cls_digit
  elif dstgsh_met == 'jac':
    return 40 + cls_digit
  elif dstgsh_met == 'tan':
    return 50 + cls_digit
  elif dstgsh_met == 'cos':
    return 60 + cls_digit
  else:
    return 70 + cls_digit
  


def plot_result(model, experiment, y_or_d, frac, microscope_h=[], microscope_m=[]):
  # figure(figsize=(8, 7), dpi=100)
  maxresult = 0
  dstgsh_idx = 5
  cls_idx = 20
  tr_cnt = 0

  train_hit_result = 0
  train_mrr_result = 0



  if model == 'narm':
    result_path = f'/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps/experiment{experiment}/result_{model}_{y_or_d}/{y_or_d[0]}{int(1/frac):03}'
  elif model == 'srgnn':
    result_path = f'/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps/experiment{experiment}/result_{model}_{y_or_d}/{y_or_d[0]}{int(1/frac):03}'
  else:
    print('model name error')
    return None

  # 폴더 안 파일들 이름 긁기
  from os import listdir
  from os.path import isfile, join
  fns = [f for f in listdir(result_path) if isfile(join(result_path, f))]
  fns = sorted(fns, key=lambda x:sort_fn(x, dstgsh_idx, cls_idx))


  only_filename = []
  for fullfn in fns:
    if fullfn[:-9] not in only_filename:  # only_filename : 뒤에 _hits.pkl, _mrrs.pkl 없는 순수이름
      only_filename.append(fullfn[:-9])
    if fullfn[dstgsh_idx:dstgsh_idx+2] == 'tr':  # train 개수 찾기
      tr_cnt += 1
  


  a = 'file name'
  b = 'hits'
  c = 'mrrs'
  print(f'{a:<50} {b:<20} \t {c:<20}')
  print()

  tr_cnt_reading = 0
  # train데이터 처리
  train_hit_result_list = []
  train_mrr_result_list = []
  
  thirdmethod_hits = []
  thirdmethod_mrrs = []
  
  # 순수 file name 돌면서 결과 프린트 ###########################################
  for onlyfn in only_filename:
    with open(f'{result_path}/{onlyfn}_hits.pkl', 'rb') as q:
      hit_result = pickle.load(q)
    with open(f'{result_path}/{onlyfn}_mrrs.pkl', 'rb') as q:
      mrr_result = pickle.load(q)


    # 빈 공간 채우기
    if model == 'srgnn':
      if len(hit_result) < 30:
        hit_result += [hit_result[-1] for _ in range(30-len(hit_result))]
      if len(mrr_result) < 30:
        mrr_result += [mrr_result[-1] for _ in range(30-len(mrr_result))]

    if model == 'narm':
      if len(hit_result) < 100:
        hit_result += [hit_result[-1] for _ in range(100-len(hit_result))]

      if len(mrr_result) < 100:
        mrr_result += [mrr_result[-1] for _ in range(100-len(mrr_result))]

    # dstgsh_idx = get_dstgsh_idx(onlyfn, y_or_d)
    # dstgsh_idx = 5

    # train데이터면 모은다
    if onlyfn[dstgsh_idx:dstgsh_idx+2] == 'tr':
      train_hit_result_list.append(hit_result)  # train 결과를 변수에 저장
      train_mrr_result_list.append(mrr_result)  # train 결과를 변수에 저장

      train_hit_result = np.mean(np.array(train_hit_result_list), axis=0)  # 결과를 평균냄
      train_mrr_result = np.mean(np.array(train_mrr_result_list), axis=0)
      tr_cnt_reading += 1

      print(f'{onlyfn:<50} {hit_result[-1]:<20.4f}  \t {mrr_result[-1]:<20.4f}')
      if tr_cnt / 2 == tr_cnt_reading:
        temp = 'mean of train : '
        print(f'{temp:<50} {train_hit_result[-1]:<20.4f}  \t {train_mrr_result[-1]:<20.4f}')
        
    # thirdmethod 모은다
    elif onlyfn[dstgsh_idx:dstgsh_idx+21] == 'thirdmethodar0.3cr0.3':
      thirdmethod_hits.append(hit_result)  # train 결과를 변수에 저장
      thirdmethod_mrrs.append(mrr_result)  # train 결과를 변수에 저장

      train_hit_result = np.mean(np.array(train_hit_result_list), axis=0)  # 결과를 평균냄
      train_mrr_result = np.mean(np.array(train_mrr_result_list), axis=0)
      tr_cnt_reading += 1

      print(f'{onlyfn:<50} {hit_result[-1]:<20.4f}  \t {mrr_result[-1]:<20.4f}')
      if tr_cnt / 2 == tr_cnt_reading:
        temp = 'mean of train : '
        print(f'{temp:<50} {train_hit_result[-1]:<20.4f}  \t {train_mrr_result[-1]:<20.4f}')

    # 아니면
    else:
      if hit_result[-1] > train_hit_result[-1]:
        hit_goodornot = f'(+{hit_result[-1] - train_hit_result[-1]:.4f})****'
      else:
        hit_goodornot = f'({hit_result[-1] - train_hit_result[-1]:.4f})'

      if mrr_result[-1] > train_mrr_result[-1]:
        mrr_goodornot = f'(+{mrr_result[-1] - train_mrr_result[-1]:.4f})****'
      else:
        mrr_goodornot = f'({mrr_result[-1] - train_mrr_result[-1]:.4f})'

      print(f'{onlyfn:50} {hit_result[-1]:<6.4f}{hit_goodornot:<10} \t {mrr_result[-1]:<6.4f}{mrr_goodornot:<10}')
  # 순수 file name 돌면서 결과 프린트 ###########################################
  
  
  # 그래프 그리기 ##############################################################
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25,10), dpi=80)
  
  # hits ############################
  tr_hit_cnt = 0
  for onlyfn in only_filename:
    with open(f'{result_path}/{onlyfn}_hits.pkl', 'rb') as q:
      hit_result = pickle.load(q)

    if onlyfn[dstgsh_idx:dstgsh_idx+2] == 'tr':
      tr_hit_cnt += 1
      if tr_hit_cnt == tr_cnt / 2:  # hits, mrrs 모두 다 가져왔으면
        axes[0].plot(train_hit_result, label=f'{y_or_d[0]}{int(1/frac)}_train', marker='o', markersize=10)
    else:
      axes[0].plot(hit_result, label=onlyfn)
      
  axes[0].set_title('Recall@20', fontsize = 22) # Y label
  axes[0].set_xlabel('epochs', fontsize = 20) # X label
  axes[0].tick_params(axis='x', labelsize=20)
  axes[0].tick_params(axis='y', labelsize=20)
  
  if microscope_h:
      axes[0].set_ylim((microscope_h[0], microscope_h[1]))


  # mrrs ############################
  tr_mrr_cnt = 0
  for onlyfn in only_filename:
    with open(f'{result_path}/{onlyfn}_mrrs.pkl', 'rb') as q:
      mrr_result = pickle.load(q)
    if onlyfn[dstgsh_idx:dstgsh_idx+2] == 'tr':
      tr_mrr_cnt += 1
      if tr_mrr_cnt == tr_cnt / 2:
        axes[1].plot(train_mrr_result, label=f'{y_or_d[0]}{int(1/frac)}_train', marker='o', markersize=10)
    else:
      axes[1].plot(mrr_result, label=onlyfn)
  axes[1].set_title('MRR@20', fontsize = 22) # Y label
  axes[1].set_xlabel('epochs', fontsize = 20) # X label
  axes[1].tick_params(axis='x', labelsize=20)
  axes[1].tick_params(axis='y', labelsize=20)
  axes[1].legend(bbox_to_anchor=(1, 1), fontsize=15)
  if microscope_m:
      axes[1].set_ylim((microscope_m[0], microscope_m[1]))
  # 그래프 그리기 ##############################################################
  
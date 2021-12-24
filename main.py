# from google.colab import drive
# drive.mount('/content/drive')

# import os
# os.chdir('/content/drive/MyDrive/015GithubRepos/da_for_sbr')

        

import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter

from preprocessing import *
from simmetric import *

experiment = 1

# dataPATH = '/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps'
# expPATH = '/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps'


dataPATH = 'G:/내 드라이브/015GithubRepos/da_for_sbr/exps'
expPATH = 'G:/내 드라이브/015GithubRepos/da_for_sbr/exps'


if not os.path.exists(f'exps/experiment{experiment}'):
    os.makedirs(f'exps/experiment{experiment}')

#%% 폴더 만들고 저장

if not os.path.exists(f'exps/experiment{experiment}/yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/diginetica'):
    os.makedirs(f'exps/experiment{experiment}/diginetica')
    
if not os.path.exists(f'exps/experiment{experiment}/result_narm_yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/result_narm_yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/result_narm_diginetica'):
    os.makedirs(f'exps/experiment{experiment}/result_narm_diginetica')
    
if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/result_srgnn_yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_diginetica'):
    os.makedirs(f'exps/experiment{experiment}/result_srgnn_diginetica')



#%%

y_or_d = 'yoochoose'  # yoochoose or diginetica

#%% 원본데이터 전처리


if y_or_d == 'yoochoose':
    filename = 'yoochoose-clicks-withHeader.dat'  # 원본 데이터 파일
    # tra_sess, tes_sess, sess_clicks = prep_yoochoose(dataPATH, filename)
    fracs = [1/64, 1/128, 1/256, 1/512]
    
else:
    filename = 'train-item-views.csv'  # 원본 데이터 파일
    # tra_sess, tes_sess, sess_clicks = prep_diginetica(dataPATH, filename)
    fracs = [1/1, 1/3, 1/6, 1/12]

# 원본데이터 전처리(통계량 확인차)
print(f'원본데이터 통계량')
# tra_ids_ori, tra_dates_ori, tra_seqs_ori, item_dict_ori = obtian_tra(sess_clicks, tra_sess)




#%% train 데이터로 저장 (마지막 아이템 떼서 라벨로)

for f in fracs:
    split_frac = int(len(tra_sess) * f)
    tra_sess_frac = tra_sess[-split_frac:]
    
    print(f'----------------------------------------------')
    print(f'frac : 1/{int(1/f)}')
    
    # frac tra_seqs 저장
    tra_ids_frac, tra_dates_frac, tra_seqs_frac, item_dict_frac = obtian_tra(sess_clicks, tra_sess_frac)
    tes_ids, tes_dates, tes_seqs = obtian_tes(sess_clicks, tes_sess, item_dict_frac)
    save_tra_seqs_frac(experiment, y_or_d, f, tra_seqs_frac)

    # frac train, test 데이터 저장
    tr_seqs_frac, tr_dates_frac, tr_labs_frac, tr_ids_frac = process_seqs(tra_seqs_frac, tra_dates_frac)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
    tra_frac = (tr_seqs_frac, tr_labs_frac)
    tes = (te_seqs, te_labs)
    
    train_name = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/f):03}_train.txt'
    print(f'train_name : {train_name}')
    test_name = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/f):03}_test.txt'
    print(f'test_name : {test_name}')
    pickle.dump(tra_frac, open(train_name, 'wb'))
    pickle.dump(tes, open(test_name, 'wb'))

#%% 
from preprocessing import *

frac = 1/64

# tra_seqs_frac 불러오기
tra_seqs_frac = load_tra_seqs_frac(experiment, y_or_d, frac)

# 아이템의 출현 횟수 확인하기
allsess = []
for s in tra_seqs_frac:
  allsess += s

allaprcnt = len(allsess)
print(f'모든 세션의 아이템 출현 수 : {allaprcnt}')

# 아이템별 출현 횟수 카운트
allitemcntr = Counter(allsess)

# 아이템의 개수 구하기
nof_items = len(allitemcntr)

print(f'총 아이템 수 : {nof_items}')


#%% 유사도지표 만들어 저장

from simmetric import *

co_mat_win = get_co_matrix_win(tra_seqs_frac, 5, True, True)
save_mat_as_csr(co_mat_win, 'coo_mat', experiment, y_or_d, frac)

pmi_mat = ppmi(co_mat_win, True)
save_mat_as_csr(pmi_mat, 'pmi_mat', experiment, y_or_d, frac)

jac_mat = get_jaccard_mat(tra_seqs_frac, 5, diag_freq = True, remove_dup = True)
save_mat_as_csr(jac_mat, 'jac_mat', experiment, y_or_d, frac)

tan_mat = get_tanimoto(tra_seqs_frac, window_size=5, diag_freq=True, remove_dup=True)
save_mat_as_csr(tan_mat, 'tan_mat', experiment, y_or_d, frac)

cos_mat_win = get_cosine(co_mat_win)
save_mat_as_csr(cos_mat_win, 'cos_mat', experiment, y_or_d, frac)

w2v_mat = get_w2v_model(tra_seqs_frac, nof_items)
save_mat_as_csr(w2v_mat, 'w2v_mat', experiment, y_or_d, frac)


#%% most_dim_d 만들기 (한번만 실행)
from simmetric import *
make_save_msd('coo', experiment, y_or_d, frac)
make_save_msd('pmi', experiment, y_or_d, frac)
make_save_msd('jac', experiment, y_or_d, frac)
make_save_msd('tan', experiment, y_or_d, frac)
make_save_msd('cos', experiment, y_or_d, frac)
make_save_msd('w2v', experiment, y_or_d, frac)

#%% 

import math
falog10 = [math.log10(_) for _ in allitemcntr.values()]
plt.hist(allitemcntr.values())
plt.hist(falog10)


coo_msd = load_msd('coo', experiment, y_or_d, frac)












